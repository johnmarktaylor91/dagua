"""Regression tests for the preregistered P5 fitting harness."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch

from dagua.eval.ruler_v4.fit import (
    FitPair,
    FittingPlan,
    JNDFitConfig,
    JudgmentRow,
    OptimizerConfig,
    PairwiseObjective,
    SceneRescorer,
    SplitPurpose,
    WeightParameter,
    fit_jnd_heterogeneity,
    fit_weights,
    load_bank,
    partition_holdouts,
)
from dagua.eval.ruler_v4.fit import (
    TestHoldoutConsumedError as HoldoutConsumedError,
)
from dagua.eval.ruler_v4.fit import (
    TestHoldoutGuard as HoldoutGuard,
)
from dagua.eval.ruler_v4.scene import Scene
from tests.eval.ruler_v4.test_score import _complete_table, _profiles, _scorable_scene

_RESEARCH_ROOT = Path.home() / ".claude/research/dagua/ruler_v4/p3"


def _weight_parameters() -> tuple[WeightParameter, WeightParameter]:
    """Build the two-parameter synthetic recovery plan.

    Returns
    -------
    tuple[WeightParameter, WeightParameter]
        Two legal universal outer weights.
    """

    return (
        WeightParameter(
            "w_structure",
            "universal",
            1.0,
            {"U01.headline": 1.0},
            ("U01",),
            lower=0.25,
            upper=4.0,
        ),
        WeightParameter(
            "w_neighborhood",
            "universal",
            1.0,
            {"U03.r_1": 1.0},
            ("U03",),
            lower=0.25,
            upper=4.0,
        ),
    )


def _synthetic_recovery_rows(count: int = 1500) -> tuple[FitPair, ...]:
    """Sample judgments from known P-mean weights.

    Parameters
    ----------
    count : int, default=1500
        Number of synthetic judgments.

    Returns
    -------
    tuple[FitPair, ...]
        Reproducible A/tie/B judgments with ground truth ``(0.6, 1.6)``.
    """

    generator = np.random.default_rng(123)
    truth = np.asarray((0.6, 1.6), dtype=np.float64)
    rows = []
    for index in range(count):
        numerator_a = generator.uniform(0.0, 4.0, size=2)
        numerator_b = generator.uniform(0.0, 4.0, size=2)
        fixed_a = float(generator.uniform(0.0, 4.0))
        fixed_b = float(generator.uniform(0.0, 4.0))
        mass = 1.0 + float(truth.sum())
        difference = (fixed_a + float(numerator_a @ truth)) / mass
        difference -= (fixed_b + float(numerator_b @ truth)) / mass
        jnd = 0.18
        probability_a = 1.0 / (1.0 + math.exp(jnd + difference))
        upper = 1.0 / (1.0 + math.exp(difference - jnd))
        probabilities = (probability_a, upper - probability_a, 1.0 - upper)
        outcome = int(generator.choice((-1, 0, 1), p=probabilities))
        rows.append(
            FitPair(
                numerator_a=tuple(numerator_a),
                numerator_b=tuple(numerator_b),
                mass_coefficients=(1.0, 1.0),
                outcome=outcome,
                fixed_numerator_a=fixed_a,
                fixed_numerator_b=fixed_b,
                fixed_mass=1.0,
                jnd=jnd,
                graph_hash=f"synthetic-{index}",
            )
        )
    return tuple(rows)


def _judgment(purpose: SplitPurpose, suffix: str = "0") -> JudgmentRow:
    """Build one minimal joined bank row for holdout and bridge tests.

    Parameters
    ----------
    purpose : SplitPurpose
        Frozen A15 consumption purpose.
    suffix : str, default="0"
        Identity suffix.

    Returns
    -------
    JudgmentRow
        Valid synthetic bank row.
    """

    role = {
        SplitPurpose.FIT: "train",
        SplitPurpose.VALIDATE: "within-family-calibration",
        SplitPurpose.TEST: "cross-family-sealed",
        SplitPurpose.DIAGNOSTIC: "adversarial",
    }[purpose]
    return JudgmentRow(
        presentation_id=f"presentation-{suffix}",
        session_id="session",
        base_pair_id=f"pair-{suffix}",
        graph_hash="graph",
        blind_id_a="A",
        blind_id_b="B",
        instrument_hash="instrument",
        era="CF@4",
        observation_profile="profile",
        verdict=1,
        tie=False,
        is_replication=False,
        role=role,
        purpose=purpose,
        primary_class="class",
        size_band="band",
        generator_family="family",
        source_path="fixture.jsonl",
    )


def test_synthetic_judgments_recover_known_weights_deterministically() -> None:
    """The fitting loop recovers both known outer weights within 0.15."""

    plan = FittingPlan(_weight_parameters())
    objective = PairwiseObjective(_synthetic_recovery_rows(), plan)
    config = OptimizerConfig(
        seed=20260811,
        steps=800,
        learning_rate=0.03,
        tolerance=1.0e-10,
        patience=150,
    )

    first = fit_weights(objective, config)
    second = fit_weights(objective, config)

    assert first == second
    assert first.weights["w_structure"] == pytest.approx(0.6, abs=0.15)
    assert first.weights["w_neighborhood"] == pytest.approx(1.6, abs=0.15)
    assert first.losses[-1] < first.losses[0]


def test_fitting_plan_refuses_off_ledger_dof_and_diag_facets() -> None:
    """The plan refuses bucket overflow, reserved buckets, and DIAG weights."""

    parameters = tuple(
        WeightParameter(
            f"w_{index}",
            "universal",
            1.0,
            {f"term_{index}": 1.0},
            ("U01",),
        )
        for index in range(10)
    )
    with pytest.raises(ValueError, match="allocation buckets exceeded"):
        FittingPlan(parameters)
    with pytest.raises(ValueError, match="assignable"):
        WeightParameter("bad", "unspent", 1.0, {"term": 1.0}, ("U01",))
    with pytest.raises(ValueError, match="DIAG"):
        WeightParameter("diag", "universal", 1.0, {"U20a.i": 1.0}, ("U20a",))


def test_fitting_plan_enforces_traceability_prior_floor() -> None:
    """A U12 fit cannot project below its preregistered prior floor."""

    parameter = WeightParameter(
        "w_u12",
        "universal",
        1.0,
        {"U12.headline": 1.0},
        ("U12",),
        lower=0.25,
    )
    with pytest.raises(ValueError, match="prior floor"):
        FittingPlan((parameter,), prior_floors={"U12": 0.5})


def test_test_holdout_access_fails_closed_on_repeat(tmp_path: Path) -> None:
    """TEST labels are returned once and stay spent across guard instances."""

    rows = tuple(_judgment(purpose, str(index)) for index, purpose in enumerate(SplitPurpose))
    partitions = partition_holdouts(rows)
    record = tmp_path / "test-access.json"

    first = HoldoutGuard(record)
    consumed = first.consume(partitions)

    assert len(consumed) == 1
    assert consumed[0].purpose is SplitPurpose.TEST
    with pytest.raises(HoldoutConsumedError):
        first.consume(partitions)
    with pytest.raises(HoldoutConsumedError):
        HoldoutGuard(record).consume(partitions)


def test_scene_rescorer_uses_score_type_m_and_caches(semantic_scene: Scene) -> None:
    """The bridge obtains subterms through score() and caches shared drawings."""

    scene = replace(_scorable_scene(semantic_scene), graph_hash="graph")
    calls: Dict[str, int] = {}

    def resolve(blind_id: str, judgment: JudgmentRow) -> Scene:
        """Resolve either displayed side to the scorer fixture.

        Parameters
        ----------
        blind_id : str
            Displayed drawing id.
        judgment : JudgmentRow
            Joined judgment context.

        Returns
        -------
        Scene
            Validated fixture scene.
        """

        del judgment
        calls[blind_id] = calls.get(blind_id, 0) + 1
        return scene

    rescorer = SceneRescorer(
        resolve,
        {"profile": _complete_table()},
        {"profile": _profiles()},
    )

    first = rescorer.score_pair(_judgment(SplitPurpose.FIT))
    second = rescorer.score_pair(_judgment(SplitPurpose.FIT, "1"))

    assert first.side_a.measurement_version == "measurement-test"
    assert "U01.headline" in first.side_a.subterms
    assert first.side_a.subterms == second.side_a.subterms
    assert calls == {"A": 1, "B": 1}


def test_jnd_heterogeneity_rejects_non_replication_rows() -> None:
    """Anti-farming prevents ordinary score dispersion from fitting JND."""

    plan = FittingPlan(_weight_parameters())
    rows = _synthetic_recovery_rows(count=4)
    with pytest.raises(ValueError, match="replication"):
        fit_jnd_heterogeneity(
            rows,
            plan,
            {"w_structure": 0.6, "w_neighborhood": 1.6},
            JNDFitConfig(steps=1),
        )


@pytest.mark.skipif(not _RESEARCH_ROOT.is_dir(), reason="frozen RULER workspace unavailable")
def test_real_main_bank_loader_and_one_step_objective_smoke() -> None:
    """At least 64 accepted real sessions survive loading and one fit step."""

    bank = load_bank(
        (_RESEARCH_ROOT / "bank/main",),
        (_RESEARCH_ROOT / "stage/maincamp/sessions",),
        _RESEARCH_ROOT / "frozen/A15_FAMILY_MAP.json",
        era="CF@4",
    )

    assert bank.report.excluded_unscheduled == 0
    assert len({row.session_id for row in bank.rows}) >= 64
    assert bank.guarded_test_count > 0
    assert all(row.purpose is not SplitPurpose.TEST for row in bank.rows)

    # Hash-only features exercise the real labels and row strata without
    # performing the prohibited P5 real-scene fit. They carry no fit result.
    smoke_rows = []
    for row in bank.rows[:128]:
        digest = hashlib.sha256(row.base_pair_id.encode("utf-8")).digest()
        numerator_a = (0.1 + digest[0] / 64.0, 0.1 + digest[1] / 64.0)
        numerator_b = (0.1 + digest[2] / 64.0, 0.1 + digest[3] / 64.0)
        smoke_rows.append(
            FitPair(
                numerator_a=numerator_a,
                numerator_b=numerator_b,
                mass_coefficients=(1.0, 1.0),
                outcome=row.outcome,
                fixed_numerator_a=0.1 + digest[4] / 64.0,
                fixed_numerator_b=0.1 + digest[5] / 64.0,
                fixed_mass=1.0,
                jnd=0.18,
                primary_class=row.primary_class,
                size_band=row.size_band,
                graph_hash=row.graph_hash,
                generator_family=row.generator_family,
                era=row.era,
                instrument_hash=row.instrument_hash,
            )
        )
    plan = FittingPlan(_weight_parameters())
    objective = PairwiseObjective(smoke_rows, plan)
    initial = objective.loss(torch.tensor((1.0, 1.0), dtype=torch.float64))
    result = fit_weights(
        objective,
        OptimizerConfig(seed=20260811, steps=1, learning_rate=0.01, patience=2),
    )

    assert torch.isfinite(initial)
    assert result.steps_completed == 1
    assert all(math.isfinite(value) for value in result.weights.values())
