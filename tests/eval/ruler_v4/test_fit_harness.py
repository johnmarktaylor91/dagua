"""Regression tests for the preregistered P5 fitting harness."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch

import dagua.eval.ruler_v4.fit.holdout as holdout_module
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
    fit_pairs_from_rescoring,
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
from dagua.eval.ruler_v4.weight_table import WeightTable
from tests.eval.ruler_v4.test_score import _complete_table, _profiles, _scorable_scene


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


def _fitted_weight_table() -> WeightTable:
    """Build a complete table whose fitted declaration matches the test plan.

    Returns
    -------
    WeightTable
        Contract-valid table with two fitted outer-weight identities.
    """

    table = _complete_table()
    identities = {
        "U01.headline": "w_structure",
        "U03.r_1": "w_neighborhood",
    }
    entries = tuple(
        replace(
            entry,
            fitted_parameter=identities[entry.subterm_id],
            provenance_class="fitted",
        )
        if entry.subterm_id in identities
        else entry
        for entry in table.entries
    )
    return replace(
        table,
        entries=entries,
        fitted_parameter_buckets={name: "universal" for name in identities.values()},
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
        confidence=2,
        is_replication=False,
        role=role,
        purpose=purpose,
        primary_class="class",
        size_band="band",
        generator_family="family",
        source_path="fixture.jsonl",
    )


def _loaded_holdout_fixture(tmp_path: Path) -> tuple[object, Path, Path, Path]:
    """Build a minimal file-backed bank with one FIT and one TEST row.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated fixture directory.

    Returns
    -------
    tuple[object, pathlib.Path, pathlib.Path, pathlib.Path]
        Loaded bank and its bank, schedule, and family-map paths.
    """

    bank_path = tmp_path / "bank.jsonl"
    schedule_path = tmp_path / "schedule.jsonl"
    family_path = tmp_path / "A15_FAMILY_MAP.json"
    graphs = {
        "fit-graph": {
            "role": "train",
            "primary_class": "class",
            "size_band": "band",
            "generator_family": "family",
        },
        "test-graph": {
            "role": "cross-family-sealed",
            "primary_class": "class",
            "size_band": "band",
            "generator_family": "family",
        },
    }
    family_path.write_text(json.dumps({"graphs": graphs}), encoding="utf-8")
    schedule_rows = []
    bank_rows = []
    for suffix, graph_hash in (("fit", "fit-graph"), ("test", "test-graph")):
        schedule_rows.append(
            {
                "presentation_id": f"presentation-{suffix}",
                "session_id": f"session-{suffix}",
                "base_pair_id": f"pair-{suffix}",
                "graph_hash": graph_hash,
                "blind_id_A": f"A-{suffix}",
                "blind_id_B": f"B-{suffix}",
                "profile_opaque_id": "profile",
                "budget_line": "PRIMARY",
            }
        )
        bank_rows.append(
            {
                "presentation_id": f"presentation-{suffix}",
                "session_id": f"session-{suffix}",
                "base_pair_id": f"pair-{suffix}",
                "graph_hash": graph_hash,
                "session_accepted": True,
                "instrument_hash": "instrument",
                "judge_id": "judge/CF@4",
                "verdict": 3 if suffix == "test" else -1,
                "tie": False,
                "confidence": 2,
                "side_bit": 0,
            }
        )
    schedule_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in schedule_rows), encoding="utf-8"
    )
    bank_path.write_text("".join(f"{json.dumps(row)}\n" for row in bank_rows), encoding="utf-8")
    bank = load_bank((bank_path,), (schedule_path,), family_path)
    return bank, bank_path, schedule_path, family_path


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


def test_prior_penalty_scales_as_one_dataset_prior_not_per_row() -> None:
    """A fixed prior contribution vanishes relative to growing evidence."""

    rows = _synthetic_recovery_rows(count=8)
    plan = FittingPlan(_weight_parameters(), prior_strength=0.1)
    objective = PairwiseObjective(rows, plan)
    weights = torch.tensor((0.5, 2.0), dtype=torch.float64)
    penalty = objective.loss(weights) - objective.negative_log_likelihood(weights)
    expected_sum = sum((math.log(value) / math.log(4.0)) ** 2 for value in (0.5, 2.0))

    assert float(penalty) == pytest.approx(0.1 * expected_sum / len(rows))


def test_objective_refuses_silent_graded_verdict_collapse() -> None:
    """A 7-point A13 response cannot silently enter the three-way model."""

    row = replace(_synthetic_recovery_rows(count=1)[0], outcome=1, graded_verdict=3)
    with pytest.raises(ValueError, match="ordered-probit"):
        PairwiseObjective((row,), FittingPlan(_weight_parameters()))


def test_objective_refuses_unidentified_scale_and_flags_bound_weights() -> None:
    """Scale-invariant fits fail and projected bound endpoints are published."""

    unidentified = FitPair(
        numerator_a=(1.0, 2.0),
        numerator_b=(2.0, 1.0),
        mass_coefficients=(1.0, 1.0),
        outcome=1,
    )
    with pytest.raises(ValueError, match="not identifiable"):
        PairwiseObjective((unidentified,), FittingPlan(_weight_parameters()))

    fixed_parameter = WeightParameter(
        "w_fixed",
        "universal",
        1.0,
        {"synthetic": 1.0},
        ("synthetic",),
        lower=1.0,
        upper=1.0,
    )
    bounded_row = FitPair(
        numerator_a=(1.0,),
        numerator_b=(2.0,),
        mass_coefficients=(1.0,),
        outcome=1,
        fixed_numerator_a=1.0,
        fixed_numerator_b=1.0,
        fixed_mass=1.0,
    )
    result = fit_weights(
        PairwiseObjective((bounded_row,), FittingPlan((fixed_parameter,))),
        OptimizerConfig(steps=1),
    )
    assert result.at_bounds == {"w_fixed": "fixed"}


def test_fitting_plan_refuses_off_ledger_dof_and_diag_facets() -> None:
    """The plan refuses bucket overflow, reserved buckets, and DIAG weights."""

    parameters = tuple(
        WeightParameter(
            f"w_{index}",
            "universal",
            1.0,
            {f"term_{index}": 1.0},
            (f"synthetic-{index}",),
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
    with pytest.raises(ValueError, match="require prior floors"):
        FittingPlan((replace(parameter, lower=0.6),))
    diluted = replace(parameter, subterm_coefficients={"U12.headline": 0.05}, lower=0.6)
    with pytest.raises(ValueError, match="prior floor"):
        FittingPlan((diluted,), prior_floors={"U12": 0.5})


def test_rescoring_bridge_reconciles_facet_and_fitted_dof_ownership() -> None:
    """Frozen table ownership defeats DIAG smuggling and identity mislabelling."""

    table = _fitted_weight_table()
    plan = FittingPlan(_weight_parameters())
    assert fit_pairs_from_rescoring((), plan, table, 2.0, {}, 0.0) == ()

    smuggled = WeightParameter("w_smuggled", "universal", 1.0, {"U20a.i": 1.0}, ("U01",))
    with pytest.raises(ValueError, match="diagnostic or weight-0"):
        fit_pairs_from_rescoring((), FittingPlan((smuggled,)), _complete_table(), 2.0, {}, 0.0)

    wrong_owner = replace(_weight_parameters()[0], facet_ids=("U99",))
    with pytest.raises(ValueError, match="belongs to U01"):
        fit_pairs_from_rescoring(
            (), FittingPlan((wrong_owner,)), _fitted_weight_table(), 2.0, {}, 0.0
        )

    duplicate_owner = replace(_weight_parameters()[1], facet_ids=("U01",))
    with pytest.raises(ValueError, match="only one fitted scalar"):
        FittingPlan((_weight_parameters()[0], duplicate_owner))


def test_test_holdout_access_fails_closed_on_all_review_defeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TEST labels stay opaque and content-bound across every review attack."""

    monkeypatch.setattr(holdout_module, "_TEST_HOLDOUT_STATE_ROOT", tmp_path / "state")
    bank, bank_path, schedule_path, family_path = _loaded_holdout_fixture(tmp_path)
    partitions = partition_holdouts(bank)

    assert bank.guarded_test_count == 1
    assert not hasattr(partitions, "_test")
    assert "verdict=3" not in repr(bank)
    with pytest.raises(TypeError):
        pickle.dumps(partitions)
    with pytest.raises(TypeError):
        HoldoutGuard(tmp_path / "alternate-record.json")

    first = HoldoutGuard()
    consumed = first.consume(partitions)

    assert len(consumed) == 1
    assert consumed[0].purpose is SplitPurpose.TEST
    assert consumed[0].verdict == 3
    with pytest.raises(HoldoutConsumedError):
        first.consume(partitions)
    with pytest.raises(HoldoutConsumedError):
        HoldoutGuard().consume(partitions)

    reloaded = load_bank((bank_path,), (schedule_path,), family_path)
    with pytest.raises(HoldoutConsumedError):
        HoldoutGuard().consume(partition_holdouts(reloaded))


def test_bank_loader_denies_pilot_and_sealed_subtrees(tmp_path: Path) -> None:
    """Public ingestion refuses direct and recursive quarantined-bank reads."""

    bank_root = tmp_path / "p3/bank"
    pilot = bank_root / "pilot"
    sealed = bank_root / "sealed"
    main = bank_root / "main"
    pilot.mkdir(parents=True)
    sealed.mkdir()
    main.mkdir()
    (pilot / "judgments.jsonl").write_text("{}\n", encoding="utf-8")
    schedule = tmp_path / "schedule.jsonl"
    family = tmp_path / "family.json"
    schedule.write_text("", encoding="utf-8")
    family.write_text('{"graphs": {}}', encoding="utf-8")

    with pytest.raises(PermissionError, match="quarantined"):
        load_bank((pilot,), (schedule,), family)
    with pytest.raises(PermissionError, match="quarantined"):
        load_bank((sealed,), (schedule,), family)
    with pytest.raises(PermissionError, match="recursive quarantined"):
        load_bank((bank_root,), (schedule,), family)


def test_bank_loader_validates_a13_labels_and_schedule_schema(tmp_path: Path) -> None:
    """Implicit abstains and contradictory labels fail closed at ingestion."""

    _, bank_path, schedule_path, family_path = _loaded_holdout_fixture(tmp_path)
    rows = [json.loads(line) for line in bank_path.read_text(encoding="utf-8").splitlines()]
    rows[0].update({"verdict": 0, "tie": False})
    bank_path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")
    implicit_abstain = load_bank((bank_path,), (schedule_path,), family_path)
    assert implicit_abstain.report.excluded_invalid == 1

    rows[0].update({"verdict": 2, "tie": True})
    bank_path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="verdict and tie"):
        load_bank((bank_path,), (schedule_path,), family_path)

    schedule_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks presentation identities"):
        load_bank((bank_path,), (schedule_path,), family_path)


def test_weight_fit_refuses_nonfit_and_replication_rows() -> None:
    """Purpose and replication provenance are enforced at the fit boundary."""

    plan = FittingPlan(_weight_parameters())
    row = _synthetic_recovery_rows(count=1)[0]
    config = OptimizerConfig(steps=1)
    with pytest.raises(ValueError, match="FIT rows only"):
        fit_weights(PairwiseObjective((replace(row, purpose=SplitPurpose.VALIDATE),), plan), config)
    with pytest.raises(ValueError, match="replication"):
        fit_weights(PairwiseObjective((replace(row, is_replication=True),), plan), config)


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
            JNDFitConfig(minimum_cell_count=2, steps=1),
        )


def test_jnd_heterogeneity_requires_actual_cross_session_side_swaps() -> None:
    """Replication flags cannot substitute for repeated side-swapped pairs."""

    plan = FittingPlan(_weight_parameters())
    source = _synthetic_recovery_rows(count=2)
    unswapped = tuple(
        replace(
            row,
            is_replication=True,
            base_pair_id=f"unique-{index}",
            session_id=f"session-{index}",
        )
        for index, row in enumerate(source)
    )
    with pytest.raises(ValueError, match="cross-session replication"):
        fit_jnd_heterogeneity(
            unswapped,
            plan,
            {"w_structure": 0.6, "w_neighborhood": 1.6},
            JNDFitConfig(minimum_cell_count=2, steps=1),
        )

    swapped = (
        replace(
            source[0],
            is_replication=True,
            base_pair_id="shared",
            session_id="session-a",
            blind_id_a="drawing-a",
            blind_id_b="drawing-b",
        ),
        replace(
            source[1],
            is_replication=True,
            base_pair_id="shared",
            session_id="session-b",
            blind_id_a="drawing-b",
            blind_id_b="drawing-a",
        ),
    )
    with pytest.raises(NotImplementedError, match="split-half stability"):
        fit_jnd_heterogeneity(
            swapped,
            plan,
            {"w_structure": 0.6, "w_neighborhood": 1.6},
            JNDFitConfig(minimum_cell_count=2, steps=1),
        )
