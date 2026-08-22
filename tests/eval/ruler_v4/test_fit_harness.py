"""Regression tests for the preregistered P5 fitting harness."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import pickle
import random
import runpy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Mapping, Optional
from unittest.mock import patch

import numpy as np
import pytest
import torch

import dagua.eval.ruler_v4.fit.access as access_module
import dagua.eval.ruler_v4.fit.bank as bank_module
import dagua.eval.ruler_v4.fit.driver as driver_module
import dagua.eval.ruler_v4.fit.optimize as optimize_module
import dagua.eval.ruler_v4.fit.uncertainty as uncertainty_module
from dagua.eval.ruler_v4.fit import (
    W08_LEDGER_KEY,
    AccessLedger,
    CalibrationLookConsumedError,
    CalibrationLookGuard,
    FitDriverConfig,
    FitPair,
    FitStartConditionError,
    FittingPlan,
    HalfAssignment,
    JNDFitConfig,
    JudgmentRow,
    OptimizerConfig,
    PairwiseObjective,
    RealFitStartConditions,
    ReusableJudgmentGuard,
    SceneRescorer,
    SideSwapAuditRow,
    SplitPurpose,
    WeightParameter,
    apply_outer_weight_split_half,
    evaluate_h_jnd_branch,
    fit_jnd_heterogeneity,
    fit_pairs_from_rescoring,
    fit_weights,
    jnd_band_calibration,
    load_bank,
    load_half_assignment,
    ordered_response_calibration,
    partition_fit_ord_lines,
    partition_holdouts,
    run_freeze1_fit,
    side_swap_audit,
    synthetic_fit_pair,
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
    """Sample seven-point probit judgments from known P-mean weights.

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
        cutpoints = np.asarray((-3, -2, -1, 1, 2, 3), dtype=np.float64) * jnd
        cdf = np.asarray(
            [
                0.5 * (1.0 + math.erf((cutpoint - difference) / math.sqrt(2.0)))
                for cutpoint in cutpoints
            ]
        )
        probabilities = np.diff(np.concatenate(([0.0], cdf, [1.0])))
        lapse = 1.0 / 109.0
        probabilities = (1.0 - lapse) * probabilities + lapse / 7.0
        verdict = int(generator.choice(tuple(range(-3, 4)), p=probabilities))
        rows.append(
            synthetic_fit_pair(
                numerator_a=tuple(numerator_a),
                numerator_b=tuple(numerator_b),
                mass_coefficients=(1.0, 1.0),
                graded_verdict=verdict,
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
        SplitPurpose.REUSABLE_HOLDOUT: "entire-class-holdout",
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
        replicate_group_id=f"pair-{suffix}",
        role=role,
        purpose=purpose,
        primary_class="class",
        size_band="band",
        generator_family="family",
        source_path="fixture.jsonl",
    )


def _jnd_success_rows(
    cell_jnds: Optional[Mapping[tuple[str, str], float]] = None,
) -> tuple[FitPair, ...]:
    """Build four supported cells of true cross-session side swaps.

    Parameters
    ----------
    cell_jnds : mapping[tuple[str, str], float] or None, optional
        Optional cell-specific JND truths used to resample the judgments.

    Returns
    -------
    tuple[FitPair, ...]
        Two presentations for each of 50 base pairs in four cells.
    """

    source = _synthetic_recovery_rows(count=200)
    generator = np.random.default_rng(20260821)
    truth = np.asarray((0.6, 1.6), dtype=np.float64)
    rows = []
    cells = (
        ("class-1", "band-1"),
        ("class-1", "band-2"),
        ("class-2", "band-1"),
        ("class-2", "band-2"),
    )
    for cell_index, cell in enumerate(cells):
        for pair_index in range(50):
            original = source[cell_index * 50 + pair_index]
            jnd = 0.18 if cell_jnds is None else cell_jnds[cell]
            if cell_jnds is None:
                verdict = original.graded_verdict
            else:
                mass = original.fixed_mass + float(np.asarray(original.mass_coefficients) @ truth)
                difference = (
                    original.fixed_numerator_a + float(np.asarray(original.numerator_a) @ truth)
                ) / mass
                difference -= (
                    original.fixed_numerator_b + float(np.asarray(original.numerator_b) @ truth)
                ) / mass
                cutpoints = np.asarray((-3, -2, -1, 1, 2, 3), dtype=np.float64) * jnd
                cdf = np.asarray(
                    [
                        0.5 * (1.0 + math.erf((cutpoint - difference) / math.sqrt(2.0)))
                        for cutpoint in cutpoints
                    ]
                )
                probabilities = np.diff(np.concatenate(([0.0], cdf, [1.0])))
                verdict = int(generator.choice(tuple(range(-3, 4)), p=probabilities))
            first = replace(
                original,
                outcome=0 if verdict == 0 else 1 if verdict > 0 else -1,
                graded_verdict=verdict,
                jnd=jnd,
                primary_class=cell[0],
                size_band=cell[1],
                graph_hash=f"graph-{pair_index % 10}",
                generator_family=f"family-{pair_index % 4}",
                is_replication=True,
                base_pair_id=f"pair-{cell_index}-{pair_index}",
                replicate_group_id=f"pair-{cell_index}-{pair_index}",
                session_id=f"session-a-{cell_index}-{pair_index}",
                blind_id_a="drawing-a",
                blind_id_b="drawing-b",
            )
            second = replace(
                first,
                numerator_a=first.numerator_b,
                numerator_b=first.numerator_a,
                fixed_numerator_a=first.fixed_numerator_b,
                fixed_numerator_b=first.fixed_numerator_a,
                outcome=-first.outcome,
                graded_verdict=-verdict,
                session_id=f"session-b-{cell_index}-{pair_index}",
                blind_id_a="drawing-b",
                blind_id_b="drawing-a",
            )
            rows.extend((first, second))
    return tuple(rows)


def _load_synthetic_bank(
    bank_inputs: tuple[Path, ...],
    schedule_inputs: tuple[Path, ...],
    family_map_path: Path,
    frozen_schedule_path: Path,
    era: Optional[str] = None,
    instrument_hash: Optional[str] = None,
) -> bank_module.JudgmentBank:
    """Load a synthetic bank through the non-campaign digest seam.

    Parameters
    ----------
    bank_inputs : tuple[pathlib.Path, ...]
        Synthetic bank JSONL paths.
    schedule_inputs : tuple[pathlib.Path, ...]
        Synthetic delivered-session manifest paths.
    family_map_path : pathlib.Path
        Synthetic A15 family map.
    frozen_schedule_path : pathlib.Path
        Synthetic frozen A16 schedule.
    era : str or None
        Optional exact era selector.
    instrument_hash : str or None
        Optional exact instrument selector.

    Returns
    -------
    JudgmentBank
        Loaded synthetic bank.

    Raises
    ------
    AssertionError
        If a test attempts to override the real campaign role hash.
    """

    family = json.loads(family_map_path.read_text(encoding="utf-8"))
    role_hash = str(family.get("role_hash", ""))
    if role_hash == bank_module._FROZEN_A15_ROLE_HASH:
        raise AssertionError("synthetic digest seam cannot target the frozen campaign role hash")
    digest = hashlib.sha256(frozen_schedule_path.read_bytes()).hexdigest()
    graphs = family.get("graphs")
    if not isinstance(graphs, dict):
        raise AssertionError("synthetic family map requires graph metadata")
    with patch.object(
        bank_module,
        "_load_a15_family_map",
        return_value=(graphs, bank_module._FROZEN_A15_ROLE_HASH),
    ):
        with patch.object(bank_module, "_FROZEN_A16_SCHEDULE_DIGEST", digest):
            return load_bank(
                bank_inputs,
                schedule_inputs,
                family_map_path,
                frozen_schedule_path,
                era=era,
                instrument_hash=instrument_hash,
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
    role_lines = "\n".join(
        f"{graph_hash}\t{graph['role']}" for graph_hash, graph in sorted(graphs.items())
    )
    role_hash = hashlib.sha256(role_lines.encode("utf-8")).hexdigest()
    family_path.write_text(json.dumps({"graphs": graphs, "role_hash": role_hash}), encoding="utf-8")
    schedule_rows = []
    bank_rows = []
    for suffix, graph_hash in (("fit", "fit-graph"), ("test", "test-graph")):
        role = graphs[graph_hash]["role"]
        schedule_rows.append(
            {
                "presentation_id": f"presentation-{suffix}",
                "session_id": f"session-{suffix}",
                "base_pair_id": f"pair-{suffix}",
                "replicate_group_id": f"pair-{suffix}",
                "graph_hash": graph_hash,
                "blind_id_A": f"A-{suffix}",
                "blind_id_B": f"B-{suffix}",
                "profile_opaque_id": "profile",
                "budget_line": "PRIMARY",
                "partition": role,
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
    bank = _load_synthetic_bank((bank_path,), (schedule_path,), family_path, schedule_path)
    return bank, bank_path, schedule_path, family_path


def test_synthetic_judgments_recover_known_weights_deterministically() -> None:
    """The fitting loop reproduces this fixed sample's MLE deterministically."""

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
    # These are sample-MLE regression pins, not claims about one draw recovering
    # population truth more tightly than its measured sampling error.
    assert first.weights["w_structure"] == pytest.approx(0.6465448, abs=1.0e-6)
    assert first.weights["w_neighborhood"] == pytest.approx(1.5967943, abs=1.0e-6)
    assert first.losses[-1] < first.losses[0]
    assert first.information_rank == 2
    assert first.condition_number >= 1.0
    assert all(
        interval[0] <= first.weights[name] <= interval[1]
        for name, interval in first.intervals.items()
    )
    narrow = replace(
        first,
        intervals={name: (value - 0.01, value + 0.01) for name, value in first.weights.items()},
    )
    half_one = replace(
        first,
        weights={"w_structure": 0.4, "w_neighborhood": 1.0},
        intervals={"w_structure": (0.25, 4.0), "w_neighborhood": (0.25, 4.0)},
    )
    half_two = replace(
        first,
        weights={"w_structure": 2.0, "w_neighborhood": 1.01},
        intervals={"w_structure": (0.25, 4.0), "w_neighborhood": (0.25, 4.0)},
    )
    stability = apply_outer_weight_split_half(narrow, half_one, half_two, plan)
    assert stability.frozen_at_prior == ("w_structure",)
    assert stability.weights["w_structure"] == 1.0


def test_weight_fit_preserves_host_rng_and_determinism_state() -> None:
    """A deterministic fit does not mutate process-global random state."""

    random.seed(19)
    np.random.seed(23)
    torch.manual_seed(29)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()

    objective = PairwiseObjective(
        _synthetic_recovery_rows(count=4), FittingPlan(_weight_parameters())
    )
    result = fit_weights(objective, OptimizerConfig(seed=20260811, steps=1))

    assert random.getstate() == python_state
    current_numpy_state = np.random.get_state()
    assert current_numpy_state[0] == numpy_state[0]
    assert np.array_equal(current_numpy_state[1], numpy_state[1])
    assert current_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert torch.are_deterministic_algorithms_enabled() is deterministic
    assert result.seed == 20260811


def test_prior_penalty_scales_as_one_dataset_prior_not_per_row() -> None:
    """Both fixed priors enter once over the shared train-row denominator."""

    rows = _synthetic_recovery_rows(count=8)
    plan = FittingPlan(_weight_parameters())
    objective = PairwiseObjective(rows, plan)
    weights = torch.tensor((0.5, 2.0), dtype=torch.float64)
    penalty = objective.loss(weights) - objective.negative_log_likelihood(weights)
    expected_sum = sum((math.log(value) / math.log(4.0)) ** 2 for value in (0.5, 2.0))
    lapse = rows[0].lapse_rate
    lapse_penalty = -math.log(lapse) - 108.0 * math.log1p(-lapse)

    assert plan.prior_strength == 2.0
    assert plan.lapse_prior_alpha == 2.0
    assert plan.lapse_prior_beta == 109.0
    assert float(penalty) == pytest.approx((2.0 * expected_sum + lapse_penalty) / len(rows))
    assert float(objective.lapse_prior_penalty()) == pytest.approx(lapse_penalty)
    assert float(objective.loss_without_lapse_prior(weights)) == pytest.approx(
        float(objective.negative_log_likelihood(weights)) + 2.0 * expected_sum / len(rows)
    )
    with pytest.raises(TypeError, match="prior_strength"):
        FittingPlan(_weight_parameters(), prior_strength=0.1)


def test_lapse_prior_is_train_only_and_real_lapse_is_one_scalar() -> None:
    """LAPSE-PRIOR never enters JND NLL and real rows cannot vary lapse by row."""

    rows = _synthetic_recovery_rows(count=4)
    plan = FittingPlan(_weight_parameters())
    objective = PairwiseObjective(rows, plan)
    weights = torch.tensor((0.6, 1.6), dtype=torch.float64)
    differences = objective.score_differences(weights).detach().numpy()
    verdicts = np.asarray([row.graded_verdict for row in rows], dtype=np.int64)
    lapses = np.asarray([row.lapse_rate for row in rows])

    jnd_nll = uncertainty_module._ordered_nll(rows[0].jnd, differences, verdicts, lapses)
    doubled_jnd_nll = uncertainty_module._ordered_nll(
        rows[0].jnd,
        np.concatenate((differences, differences)),
        np.concatenate((verdicts, verdicts)),
        np.concatenate((lapses, lapses)),
    )

    assert doubled_jnd_nll == pytest.approx(2.0 * jnd_nll)
    real_rows = tuple(replace(row, synthetic=False) for row in rows)
    with pytest.raises(ValueError, match="one broadcast lapse"):
        PairwiseObjective(
            (real_rows[0], replace(real_rows[1], lapse_rate=0.03)),
            plan,
        )


def test_exact_duplication_weakens_one_dataset_lapse_prior() -> None:
    """Duplicating evidence moves lapse toward data rather than duplicating its prior."""

    fixed = WeightParameter(
        "fixed-weight",
        "universal",
        1.0,
        {"synthetic": 1.0},
        ("U01",),
        lower=1.0,
        upper=1.0,
    )
    plan = FittingPlan((fixed,))
    outlier = synthetic_fit_pair(
        numerator_a=(0.0,),
        numerator_b=(10.0,),
        mass_coefficients=(1.0,),
        graded_verdict=3,
        fixed_mass=1.0,
    )

    _, sparse_lapse, _ = driver_module._fit_weight_lapse_block(
        (outlier,) * 2,
        plan,
        FitDriverConfig(),
    )
    _, duplicated_lapse, _ = driver_module._fit_weight_lapse_block(
        (outlier,) * 20,
        plan,
        FitDriverConfig(),
    )

    assert 1.0 / 109.0 < sparse_lapse < duplicated_lapse < 0.25


def test_lapse_boundary_disclosure_publishes_both_objectives() -> None:
    """A bound-pinned lapse names the bound and penalized/unpenalized objectives."""

    rows = tuple(replace(row, lapse_rate=0.25) for row in _synthetic_recovery_rows(count=4))
    plan = FittingPlan(_weight_parameters())
    weights = {parameter.name: parameter.prior for parameter in plan.weights}

    disclosure = driver_module._lapse_boundary_disclosure(rows, plan, weights, 0.25)

    assert disclosure is not None
    assert disclosure.bound == "upper"
    assert disclosure.fitted_value == 0.25
    assert disclosure.penalized_objective > disclosure.unpenalized_objective


def test_objective_consumes_all_seven_graded_verdicts_with_probit_cutpoints() -> None:
    """The full A13 scale reaches the fixed-ratio ordered-probit likelihood."""

    with pytest.raises(TypeError, match="graded_verdict"):
        FitPair(
            numerator_a=(1.0,),
            numerator_b=(2.0,),
            mass_coefficients=(1.0,),
            outcome=1,
        )
    rows = tuple(
        replace(
            _synthetic_recovery_rows(count=1)[0],
            outcome=0 if verdict == 0 else (1 if verdict > 0 else -1),
            graded_verdict=verdict,
            lapse_rate=0.0,
        )
        for verdict in range(-3, 4)
    )
    objective = PairwiseObjective(rows, FittingPlan(_weight_parameters()))
    weights = torch.tensor((1.0, 1.0), dtype=torch.float64)
    probabilities = objective.outcome_probabilities(weights)

    assert probabilities.shape == (7, 7)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(7, dtype=torch.float64))
    assert objective.directional_probabilities(weights).shape == (7, 3)
    difference = float(objective.score_differences(weights)[0])
    first_cutpoint = rows[0].jnd
    expected_tie = 0.5 * (
        math.erf((first_cutpoint - difference) / math.sqrt(2.0))
        - math.erf((-first_cutpoint - difference) / math.sqrt(2.0))
    )
    assert float(probabilities[0, 3]) == pytest.approx(expected_tie)
    lapsed = PairwiseObjective(
        tuple(replace(row, lapse_rate=0.35) for row in rows), objective.plan
    ).outcome_probabilities(weights)
    assert torch.allclose(lapsed, 0.65 * probabilities + 0.35 / 7.0)
    confidence_changed = PairwiseObjective(
        tuple(replace(row, confidence=3) for row in rows), objective.plan
    )
    assert torch.equal(
        probabilities,
        confidence_changed.outcome_probabilities(weights),
    )
    calibration = ordered_response_calibration(
        objective,
        {"w_structure": 1.0, "w_neighborhood": 1.0},
    )
    assert sum(item.count for item in calibration) == 7
    assert all(len(item.observed) == len(item.predicted) == 7 for item in calibration)


def test_objective_refuses_unidentified_scale_and_flags_bound_weights() -> None:
    """Scale-invariant fits fail and projected bound endpoints are published."""

    unidentified = synthetic_fit_pair(
        numerator_a=(1.0, 2.0),
        numerator_b=(2.0, 1.0),
        mass_coefficients=(1.0, 1.0),
        graded_verdict=1,
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
    bounded_row = synthetic_fit_pair(
        numerator_a=(1.0,),
        numerator_b=(2.0,),
        mass_coefficients=(1.0,),
        graded_verdict=1,
        fixed_numerator_a=1.0,
        fixed_numerator_b=1.0,
        fixed_mass=1.0,
    )
    result = fit_weights(
        PairwiseObjective((bounded_row,), FittingPlan((fixed_parameter,))),
        OptimizerConfig(steps=1),
    )
    assert result.at_bounds == {"w_fixed": "fixed"}


def test_observed_information_excludes_the_weight_prior() -> None:
    """A collinear likelihood remains rank deficient in the publication."""

    rows = tuple(
        replace(
            row,
            numerator_a=(row.numerator_a[0], row.numerator_a[0]),
            numerator_b=(row.numerator_b[0], row.numerator_b[0]),
            mass_coefficients=(1.0, 1.0),
        )
        for row in _synthetic_recovery_rows(count=40)
    )
    objective = PairwiseObjective(rows, FittingPlan(_weight_parameters()))

    rank, condition_number = optimize_module._information_diagnostics(
        objective, {"w_structure": 1.0, "w_neighborhood": 1.0}
    )

    assert rank == 1
    assert condition_number > 1.0e12


def test_jnd_calibration_replaces_only_the_dataclass_band() -> None:
    """Cell calibration preserves every provenance field while replacing JND."""

    row = _synthetic_recovery_rows(count=1)[0]
    results = jnd_band_calibration(
        (row,),
        FittingPlan(_weight_parameters()),
        {"w_structure": 0.6, "w_neighborhood": 1.6},
        {("synthetic", "synthetic"): 0.25},
    )
    assert len(results) == 1
    assert results[0].jnd == 0.25


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
    multi_facet = replace(
        parameter,
        subterm_coefficients={"U12.headline": 0.05, "U13.x": 1.0},
        facet_ids=("U12", "U13"),
        lower=0.6,
    )
    with pytest.raises(ValueError, match="one fitted scalar per traceability facet"):
        FittingPlan((multi_facet,), prior_floors={"U12": 0.5, "U13": 0.5})


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

    monkeypatch.setattr(access_module, "_ACCESS_LEDGER_ROOT", tmp_path / "state")
    bank, bank_path, schedule_path, family_path = _loaded_holdout_fixture(tmp_path)
    partitions = partition_holdouts(bank)

    assert bank.guarded_test_count == 1
    assert not hasattr(partitions, "_test")
    assert "verdict=3" not in repr(bank)
    with pytest.raises(TypeError):
        pickle.dumps(bank)
    with pytest.raises(TypeError):
        pickle.dumps(partitions)
    assert HoldoutGuard()._ledger_root == tmp_path / "state"
    with pytest.raises(TypeError):
        HoldoutGuard(tmp_path / "alternate-root")
    assert not hasattr(bank_module, "_reveal_test_rows")
    refs = partitions._test_refs_by_role["cross-family-sealed"]
    with pytest.raises(RuntimeError, match="reservation"):
        HoldoutGuard()._reveal_test_rows(refs, "cross-family-sealed")

    first = HoldoutGuard()
    consumed = first.consume(partitions, "cross-family-sealed")

    assert len(consumed) == 1
    assert consumed[0].purpose is SplitPurpose.TEST
    assert consumed[0].verdict == 3
    with pytest.raises(HoldoutConsumedError):
        first.consume(partitions, "cross-family-sealed")
    with pytest.raises(HoldoutConsumedError):
        HoldoutGuard().consume(partitions, "cross-family-sealed")

    reloaded = _load_synthetic_bank((bank_path,), (schedule_path,), family_path, schedule_path)
    with pytest.raises(HoldoutConsumedError):
        HoldoutGuard().consume(partition_holdouts(reloaded), "cross-family-sealed")


def test_test_holdout_refuses_empty_partition_before_spending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The natural ``partition_holdouts(bank.rows)`` composition cannot burn TEST."""

    monkeypatch.setattr(access_module, "_ACCESS_LEDGER_ROOT", tmp_path / "state")
    bank, _, _, _ = _loaded_holdout_fixture(tmp_path)
    empty_test = partition_holdouts(bank.rows)
    with pytest.raises(ValueError, match="empty"):
        HoldoutGuard().consume(empty_test, "cross-family-sealed")
    assert not (tmp_path / "state").exists()


def test_holdout_default_ledger_root_is_frozen_campaign_config() -> None:
    """Checkout location cannot mint an independent sealed-role budget."""

    expected = Path("/home/jtaylor/.claude/research/dagua/ruler_v4/p3/gate/ACCESS_LEDGER")

    assert access_module._ACCESS_LEDGER_ROOT == expected
    assert HoldoutGuard()._ledger_root == expected
    assert expected.is_absolute()


def test_frozen_schedule_digest_is_pinned_against_real_role_hash_attack(tmp_path: Path) -> None:
    """A crafted self-attested census cannot target the real ledger identity."""

    _, bank_path, schedule_path, family_path = _loaded_holdout_fixture(tmp_path)
    family = json.loads(family_path.read_text(encoding="utf-8"))
    crafted_digest = hashlib.sha256(schedule_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="frozen campaign identity"):
        load_bank((bank_path,), (schedule_path,), family_path, schedule_path)
    assert "frozen_schedule_digest" not in inspect.signature(load_bank).parameters
    with patch.object(
        bank_module,
        "_load_a15_family_map",
        return_value=(family["graphs"], bank_module._FROZEN_A15_ROLE_HASH),
    ):
        with patch.object(
            bank_module,
            "_TEST_ONLY_A16_DIGESTS_BY_ROLE_HASH",
            {bank_module._FROZEN_A15_ROLE_HASH: crafted_digest},
        ):
            with pytest.raises(ValueError, match="digest does not match"):
                load_bank((bank_path,), (schedule_path,), family_path, schedule_path)


def test_holdout_reconciles_replanned_ids_and_rejects_equal_size_content_attack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Base-pair content joins disjoint plan/campaign ids and defeats count swaps."""

    monkeypatch.setattr(access_module, "_ACCESS_LEDGER_ROOT", tmp_path / "state")
    _, bank_path, schedule_path, family_path = _loaded_holdout_fixture(tmp_path)
    frozen_schedule_path = tmp_path / "PRESENTATION_SCHEDULE.jsonl"
    frozen_schedule_path.write_bytes(schedule_path.read_bytes())
    schedule_rows = [
        json.loads(line) for line in schedule_path.read_text(encoding="utf-8").splitlines()
    ]
    bank_rows = [json.loads(line) for line in bank_path.read_text(encoding="utf-8").splitlines()]
    schedule_rows[1].update(
        presentation_id="campaign-presentation-test",
        session_id="main-campaign-session-test",
        base_pair_id="crafted-pair-same-cardinality",
    )
    bank_rows[1].update(
        presentation_id="campaign-presentation-test",
        session_id="main-campaign-session-test",
        base_pair_id="crafted-pair-same-cardinality",
    )
    schedule_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in schedule_rows), encoding="utf-8"
    )
    bank_path.write_text("".join(f"{json.dumps(row)}\n" for row in bank_rows), encoding="utf-8")
    attacked = partition_holdouts(
        _load_synthetic_bank((bank_path,), (schedule_path,), family_path, frozen_schedule_path)
    )

    assert len(attacked._test_refs_by_role["cross-family-sealed"]) == len(
        attacked.expected_test_base_pairs["cross-family-sealed"]
    )
    with pytest.raises(ValueError, match="not a subset"):
        HoldoutGuard().consume(attacked, "cross-family-sealed")
    assert not (tmp_path / "state").exists()

    schedule_rows[1]["base_pair_id"] = "pair-test"
    bank_rows[1]["base_pair_id"] = "pair-test"
    schedule_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in schedule_rows), encoding="utf-8"
    )
    bank_path.write_text("".join(f"{json.dumps(row)}\n" for row in bank_rows), encoding="utf-8")
    reconciled = partition_holdouts(
        _load_synthetic_bank((bank_path,), (schedule_path,), family_path, frozen_schedule_path)
    )

    assert reconciled.expected_test_base_pairs["cross-family-sealed"] == ("pair-test",)
    assert len(HoldoutGuard().consume(reconciled, "cross-family-sealed")) == 1


def test_test_holdout_identity_is_role_hash_keyed_and_refuses_partial_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loader selectors cannot mint a fresh one-shot or spend a partial seal."""

    monkeypatch.setattr(access_module, "_ACCESS_LEDGER_ROOT", tmp_path / "state")
    _, bank_path, schedule_path, family_path = _loaded_holdout_fixture(tmp_path)
    schedule_rows = [
        json.loads(line) for line in schedule_path.read_text(encoding="utf-8").splitlines()
    ]
    bank_rows = [json.loads(line) for line in bank_path.read_text(encoding="utf-8").splitlines()]
    second_schedule = dict(schedule_rows[1])
    second_schedule.update(
        presentation_id="presentation-test-cf1",
        session_id="session-test-cf1",
        base_pair_id="pair-test-cf1",
    )
    second_bank = dict(bank_rows[1])
    second_bank.update(
        presentation_id="presentation-test-cf1",
        session_id="main-session-test-cf1",
        base_pair_id="pair-test-cf1",
        judge_id="judge/CF@1",
    )
    schedule_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in (*schedule_rows, second_schedule)),
        encoding="utf-8",
    )
    bank_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in (*bank_rows, second_bank)),
        encoding="utf-8",
    )
    full = partition_holdouts(
        _load_synthetic_bank((bank_path,), (schedule_path,), family_path, schedule_path)
    )
    subset_bank = _load_synthetic_bank(
        (bank_path,),
        (schedule_path,),
        family_path,
        schedule_path,
        era="CF@1",
    )
    subset = partition_holdouts(subset_bank)

    assert full.role_hash == subset.role_hash
    with pytest.raises(ValueError, match="partial"):
        HoldoutGuard().consume(subset, "cross-family-sealed")
    assert not (tmp_path / "state").exists()

    bank_subset_path = tmp_path / "bank-subset.jsonl"
    bank_subset_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in bank_rows), encoding="utf-8"
    )
    bank_subset = partition_holdouts(
        _load_synthetic_bank(
            (bank_subset_path,),
            (schedule_path,),
            family_path,
            schedule_path,
        )
    )
    with pytest.raises(ValueError, match="partial"):
        HoldoutGuard().consume(bank_subset, "cross-family-sealed")

    schedule_subset_path = tmp_path / "schedule-subset.jsonl"
    schedule_subset_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in schedule_rows), encoding="utf-8"
    )
    schedule_subset = partition_holdouts(
        _load_synthetic_bank(
            (bank_path,),
            (schedule_subset_path,),
            family_path,
            schedule_path,
        )
    )
    with pytest.raises(ValueError, match="partial"):
        HoldoutGuard().consume(schedule_subset, "cross-family-sealed")

    consumed = HoldoutGuard().consume(full, "cross-family-sealed")
    assert len(consumed) == 2
    with pytest.raises(HoldoutConsumedError):
        HoldoutGuard().consume(full, "cross-family-sealed")


def test_entire_class_holdout_is_reusable_and_never_spends_the_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A15's reusable entire-class role remains outside the once-only guard."""

    _, bank_path, schedule_path, family_path = _loaded_holdout_fixture(tmp_path)
    family = json.loads(family_path.read_text(encoding="utf-8"))
    family["graphs"]["class-holdout-graph"] = {
        "role": "entire-class-holdout",
        "primary_class": "held-out-class",
        "size_band": "band",
        "generator_family": "held-out-family",
    }
    family["graphs"]["unbanked-class-holdout-graph"] = {
        "role": "entire-class-holdout",
        "primary_class": "held-out-class",
        "size_band": "band",
        "generator_family": "held-out-family",
    }
    role_lines = "\n".join(
        f"{graph_hash}\t{graph['role']}" for graph_hash, graph in sorted(family["graphs"].items())
    )
    family["role_hash"] = hashlib.sha256(role_lines.encode("utf-8")).hexdigest()
    family_path.write_text(json.dumps(family), encoding="utf-8")
    schedule_row = {
        "presentation_id": "presentation-class-holdout",
        "session_id": "session-class-holdout",
        "base_pair_id": "pair-class-holdout",
        "graph_hash": "class-holdout-graph",
        "blind_id_A": "A-class-holdout",
        "blind_id_B": "B-class-holdout",
        "profile_opaque_id": "profile",
        "budget_line": "PRIMARY",
        "partition": "entire-class-holdout",
    }
    unbanked_schedule_row = {
        **schedule_row,
        "presentation_id": "presentation-class-holdout-unbanked",
        "session_id": "session-class-holdout-unbanked",
        "base_pair_id": "pair-class-holdout-unbanked",
        "graph_hash": "unbanked-class-holdout-graph",
    }
    bank_row = {
        "presentation_id": "presentation-class-holdout",
        "session_id": "session-class-holdout",
        "base_pair_id": "pair-class-holdout",
        "graph_hash": "class-holdout-graph",
        "session_accepted": True,
        "instrument_hash": "instrument",
        "judge_id": "judge/CF@4",
        "verdict": 1,
        "tie": False,
        "confidence": 2,
        "side_bit": 0,
    }
    with schedule_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(schedule_row)}\n")
        handle.write(f"{json.dumps(unbanked_schedule_row)}\n")
    with bank_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(bank_row)}\n")

    bank = _load_synthetic_bank((bank_path,), (schedule_path,), family_path, schedule_path)
    partitions = partition_holdouts(bank)

    monkeypatch.setattr(access_module, "_ACCESS_LEDGER_ROOT", tmp_path / "state")
    assert bank.guarded_test_count == 1
    with pytest.raises(ValueError, match="LOOK-LEDGER"):
        bank.select(purpose=SplitPurpose.REUSABLE_HOLDOUT)
    assert len(partitions.reusable_holdout) == 1
    assert partitions.reusable_holdout[0].role == "entire-class-holdout"
    labelled = ReusableJudgmentGuard().consume(partitions, SplitPurpose.REUSABLE_HOLDOUT)
    assert labelled[0].verdict == 1
    assert not any((tmp_path / "state").glob(f"{partitions.role_hash}.cross-family-sealed.*.lock"))


def test_sealed_roles_have_separate_labelled_in_tree_ledger_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Within/cross sealed roles each append one independently budgeted record."""

    ledger_root = tmp_path / "p3/gate/ACCESS_LEDGER"
    monkeypatch.setattr(access_module, "_ACCESS_LEDGER_ROOT", ledger_root)
    _, bank_path, schedule_path, family_path = _loaded_holdout_fixture(tmp_path)
    family = json.loads(family_path.read_text(encoding="utf-8"))
    family["graphs"]["within-test-graph"] = {
        "role": "within-family-sealed",
        "primary_class": "class",
        "size_band": "band",
        "generator_family": "family",
    }
    role_lines = "\n".join(
        f"{graph_hash}\t{graph['role']}" for graph_hash, graph in sorted(family["graphs"].items())
    )
    family["role_hash"] = hashlib.sha256(role_lines.encode("utf-8")).hexdigest()
    family_path.write_text(json.dumps(family), encoding="utf-8")
    schedule_row = {
        "presentation_id": "presentation-within",
        "session_id": "session-within",
        "base_pair_id": "pair-within",
        "graph_hash": "within-test-graph",
        "blind_id_A": "A-within",
        "blind_id_B": "B-within",
        "profile_opaque_id": "profile",
        "budget_line": "PRIMARY",
        "partition": "within-family-sealed",
    }
    bank_row = {
        "presentation_id": "presentation-within",
        "session_id": "session-within",
        "base_pair_id": "pair-within",
        "graph_hash": "within-test-graph",
        "session_accepted": True,
        "instrument_hash": "instrument",
        "judge_id": "judge/CF@4",
        "verdict": -2,
        "tie": False,
        "confidence": 3,
        "side_bit": 0,
    }
    with schedule_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(schedule_row)}\n")
    with bank_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(bank_row)}\n")
    partitions = partition_holdouts(
        _load_synthetic_bank((bank_path,), (schedule_path,), family_path, schedule_path)
    )

    cross_guard = HoldoutGuard()
    cross = cross_guard.consume(partitions, "cross-family-sealed")
    with pytest.raises(RuntimeError, match="reservation"):
        cross_guard._reveal_test_rows(
            partitions._test_refs_by_role["within-family-sealed"],
            "within-family-sealed",
        )
    assert not (ledger_root / f"{partitions.role_hash}.within-family-sealed.1.lock").exists()
    within = HoldoutGuard().consume(partitions, "within-family-sealed")

    assert [row.role for row in cross] == ["cross-family-sealed"]
    assert [row.role for row in within] == ["within-family-sealed"]
    ledger_path = ledger_root / f"{partitions.role_hash}.jsonl"
    records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert [(record["ledger_key"], record["purpose"], record["budget"]) for record in records] == [
        ("cross-family-sealed", "sealed-test-cross", 1),
        ("within-family-sealed", "sealed-test-within", 1),
    ]
    with pytest.raises(HoldoutConsumedError):
        HoldoutGuard().consume(partitions, "cross-family-sealed")
    with pytest.raises(HoldoutConsumedError):
        HoldoutGuard().consume(partitions, "within-family-sealed")


def test_calibration_labels_require_ordered_alpha_spent_looks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calibration metadata is unlimited while every label release spends one slot."""

    ledger_root = tmp_path / "ACCESS_LEDGER"
    monkeypatch.setattr(access_module, "_ACCESS_LEDGER_ROOT", ledger_root)
    _, bank_path, schedule_path, family_path = _loaded_holdout_fixture(tmp_path)
    family = json.loads(family_path.read_text(encoding="utf-8"))
    family["graphs"]["calibration-graph"] = {
        "role": "within-family-calibration",
        "primary_class": "class",
        "size_band": "band",
        "generator_family": "family",
    }
    role_lines = "\n".join(
        f"{graph_hash}\t{graph['role']}" for graph_hash, graph in sorted(family["graphs"].items())
    )
    family["role_hash"] = hashlib.sha256(role_lines.encode("utf-8")).hexdigest()
    family_path.write_text(json.dumps(family), encoding="utf-8")
    schedule_row = {
        "presentation_id": "presentation-calibration",
        "session_id": "session-calibration",
        "base_pair_id": "pair-calibration",
        "graph_hash": "calibration-graph",
        "blind_id_A": "opaque-A",
        "blind_id_B": "opaque-B",
        "profile_opaque_id": "profile",
        "budget_line": "PRIMARY",
        "partition": "within-family-calibration",
    }
    bank_row = {
        "presentation_id": "presentation-calibration",
        "session_id": "session-calibration",
        "base_pair_id": "pair-calibration",
        "graph_hash": "calibration-graph",
        "session_accepted": True,
        "instrument_hash": "instrument",
        "judge_id": "judge/CF@4",
        "verdict": -3,
        "tie": False,
        "confidence": 3,
        "reason_tags": ["crossing"],
        "served_model_fingerprint": {"usage": {"tokens": 99}},
        "free_note": "gated",
        "side_bit": 0,
    }
    abstain_schedule_row = {
        **schedule_row,
        "presentation_id": "presentation-calibration-abstain",
        "session_id": "session-calibration-abstain",
        "base_pair_id": "pair-calibration-abstain",
    }
    abstain_bank_row = {
        **bank_row,
        "presentation_id": "presentation-calibration-abstain",
        "session_id": "session-calibration-abstain",
        "base_pair_id": "pair-calibration-abstain",
        "judge_id": "judge/CF@1",
        "verdict": 0,
        "tie": False,
        "abstain": True,
    }
    with schedule_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(schedule_row)}\n")
        handle.write(f"{json.dumps(abstain_schedule_row)}\n")
    with bank_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(bank_row)}\n")
        handle.write(f"{json.dumps(abstain_bank_row)}\n")
    bank = _load_synthetic_bank((bank_path,), (schedule_path,), family_path, schedule_path)
    partitions = partition_holdouts(bank)

    metadata = bank.metadata(purpose=SplitPurpose.VALIDATE)
    assert len(metadata) == 2
    assert bank.report.excluded_invalid == 0
    gated = {"verdict", "confidence", "free_note", "source_path", "blind_id_a"}
    assert not (gated & set(vars(metadata[0])))
    assert all(row.purpose is SplitPurpose.FIT for row in bank.rows)
    with pytest.raises(ValueError, match="LOOK-LEDGER"):
        bank.select(purpose=SplitPurpose.VALIDATE)
    with pytest.raises(ValueError, match="expected 'post-M1'"):
        CalibrationLookGuard().consume(
            partitions,
            "within-family-calibration",
            "post-M2",
            100,
            "capacity unlock",
        )

    filtered = partition_holdouts(
        _load_synthetic_bank(
            (bank_path,),
            (schedule_path,),
            family_path,
            schedule_path,
            era="CF@4",
        )
    )
    partial_ledger_root = tmp_path / "PARTIAL_ACCESS_LEDGER"
    monkeypatch.setattr(access_module, "_ACCESS_LEDGER_ROOT", partial_ledger_root)
    partial_rows, partial_reservation = CalibrationLookGuard().consume(
        filtered,
        "within-family-calibration",
        "post-M1",
        100,
        "capacity unlock",
    )
    assert len(partial_rows) == 1
    assert partial_reservation.slot_index == 1

    monkeypatch.setattr(access_module, "_ACCESS_LEDGER_ROOT", ledger_root)
    assert not ledger_root.exists()

    occasions = ("post-M1", "post-M2", "post-M3", "stopping")
    prior_cumulative = 0.0
    for index, occasion in enumerate(occasions, start=1):
        rows, reservation = CalibrationLookGuard().consume(
            partitions,
            "within-family-calibration",
            occasion,
            index * 100,
            "stopping" if occasion == "stopping" else "capacity unlock",
        )
        assert len(rows) == 1
        assert rows[0].verdict == -3
        assert reservation.slot_index == index
        assert reservation.information_fraction == pytest.approx(index * 100 / 8520)
        assert reservation.incremental_alpha == pytest.approx(
            float(reservation.cumulative_alpha) - prior_cumulative
        )
        prior_cumulative = float(reservation.cumulative_alpha)
    with pytest.raises(CalibrationLookConsumedError, match="exhausted"):
        CalibrationLookGuard().consume(
            partitions,
            "within-family-calibration",
            "stopping",
            500,
            "stopping",
        )

    records = [
        json.loads(line)
        for line in (ledger_root / f"{partitions.role_hash}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["occasion"] for record in records] == list(occasions)
    assert all(record["row_set_digest"] and record["decision"] for record in records)


def test_w08_look_schedule_is_separate_and_has_no_alpha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W-08 uses its own four ordered slots without alpha arithmetic."""

    monkeypatch.setattr(access_module, "_ACCESS_LEDGER_ROOT", tmp_path / "ACCESS_LEDGER")
    ledger = AccessLedger()
    first = ledger.reserve_look(
        bank_module._FROZEN_A15_ROLE_HASH,
        W08_LEDGER_KEY,
        "post-M1",
        ("morph-1", "morph-2"),
        "off-distribution agreement report",
    )

    assert first.slot_index == 1
    assert first.information_fraction is None
    assert first.cumulative_alpha is None
    assert first.incremental_alpha is None
    with pytest.raises(ValueError, match="no alpha"):
        ledger.reserve_look(
            bank_module._FROZEN_A15_ROLE_HASH,
            W08_LEDGER_KEY,
            "post-M2",
            ("morph-1",),
            "off-distribution agreement report",
            informative_judgments=100,
        )


def test_calibration_alpha_spend_matches_frozen_disclosure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calibration looks spend increasing upper-tail O'Brien-Fleming alpha."""

    monkeypatch.setattr(access_module, "_ACCESS_LEDGER_ROOT", tmp_path / "ACCESS_LEDGER")
    ledger = AccessLedger()
    role_hash = bank_module._FROZEN_A15_ROLE_HASH
    informative_counts = (2000, 4000, 6000, 7100)
    cumulative = []
    incremental = []
    for occasion, informative_count in zip(
        ("post-M1", "post-M2", "post-M3", "stopping"), informative_counts
    ):
        reservation = ledger.reserve_look(
            role_hash,
            "within-family-calibration",
            occasion,
            ("calibration-row",),
            "capacity unlock",
            informative_judgments=informative_count,
        )
        cumulative.append(float(reservation.cumulative_alpha))
        incremental.append(float(reservation.incremental_alpha))

    assert cumulative == sorted(cumulative)
    assert all(value >= 0.0 for value in incremental)
    assert cumulative[-1] == pytest.approx(0.031791, abs=5.0e-7)
    assert access_module._cumulative_alpha(0.7) == pytest.approx(0.019150, abs=5.0e-7)
    assert access_module._cumulative_alpha(1.0) == pytest.approx(0.05)


def test_ledger_and_h_jnd_refuse_caller_minted_role_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every persistent budget is keyed only by the frozen A15 identity."""

    monkeypatch.setattr(access_module, "_ACCESS_LEDGER_ROOT", tmp_path / "ACCESS_LEDGER")
    with pytest.raises(ValueError, match="frozen A15 role hash"):
        AccessLedger().reserve_once("minted", "within-family-sealed", ("row",), "sealed")
    with pytest.raises(ValueError, match="frozen A15 role hash"):
        JNDFitConfig(
            role_hash="minted",
            top_composite_pair_counts={"band": 67},
            rotation_envelopes={"class": 0.01},
        )
    assert not (tmp_path / "ACCESS_LEDGER").exists()


def test_ledger_annulment_is_append_only_and_restores_synthetic_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Annul a synthetic-only spend and reuse its generation-suffixed slot."""

    ledger_root = tmp_path / "ACCESS_LEDGER"
    ledger = AccessLedger(ledger_root)
    role_hash = bank_module._FROZEN_A15_ROLE_HASH
    ledger.reserve_once(
        role_hash,
        access_module.H_JND_LEDGER_KEY,
        ("synthetic-row",),
        "synthetic branch probe",
        synthetic_only=True,
    )
    ledger_path = ledger_root / f"{role_hash}.jsonl"
    original_line = ledger_path.read_bytes().splitlines(keepends=True)[0]
    original_digest = hashlib.sha256(original_line).hexdigest()
    original_lock = ledger_root / f"{role_hash}.{access_module.H_JND_LEDGER_KEY}.1.lock"

    with pytest.raises(ValueError, match="landed licensing addendum"):
        ledger.annul_reservation(
            role_hash,
            access_module.H_JND_LEDGER_KEY,
            1,
            "ADDENDUM-28 does not license this specific synthetic correction.",
            authority_addendum=28,
            scope_basis="synthetic_only_manifest",
        )
    monkeypatch.setattr(
        access_module,
        "_LICENSED_ANNULMENTS",
        frozenset({(29, access_module.H_JND_LEDGER_KEY, 1)}),
    )

    annulment = ledger.annul_reservation(
        role_hash,
        access_module.H_JND_LEDGER_KEY,
        1,
        "A synthetic-only probe with tau 0.5 incorrectly consumed the campaign-style slot.",
        authority_addendum=29,
        scope_basis="synthetic_only_manifest",
    )

    assert annulment.annulled_line_sha256 == original_digest
    assert original_lock.exists()
    assert ledger.budget_usage(role_hash)[access_module.H_JND_LEDGER_KEY] == 0
    assert len(ledger.annulment_lines(role_hash)) == 1
    assert ledger.ledger_defects(role_hash) == ()
    ledger.reserve_once(
        role_hash,
        access_module.H_JND_LEDGER_KEY,
        ("replacement-row",),
        "replacement synthetic branch probe",
        synthetic_only=True,
    )
    replacement_lock = ledger_root / f"{role_hash}.{access_module.H_JND_LEDGER_KEY}.1.a1.lock"
    assert replacement_lock.exists()
    assert original_lock.exists()
    assert len(ledger_path.read_bytes().splitlines()) == 3
    assert ledger.budget_usage(role_hash)[access_module.H_JND_LEDGER_KEY] == 1
    with pytest.raises(ValueError, match="landed licensing addendum"):
        ledger.annul_reservation(
            role_hash,
            access_module.H_JND_LEDGER_KEY,
            1,
            "An unlicensed correction must be refused.",
            authority_addendum=28,
            scope_basis="synthetic_only_manifest",
        )
    replacement_line = ledger_path.read_bytes().splitlines(keepends=True)[-1]
    void_annulment = {
        "state": "ANNUL",
        "role_hash": role_hash,
        "ledger_key": access_module.H_JND_LEDGER_KEY,
        "slot_index": 1,
        "annulled_line_sha256": hashlib.sha256(replacement_line).hexdigest(),
        "reason": "This forged entry cites no landed authority.",
        "authority_addendum": 28,
        "date": "2026-08-21",
        "scope_basis": "synthetic_only_manifest",
    }
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(void_annulment, sort_keys=True, separators=(",", ":")) + "\n")
    assert ledger.budget_usage(role_hash)[access_module.H_JND_LEDGER_KEY] == 1
    assert ledger.ledger_defects(role_hash) == ("line 4: ANNUL cites no landed licensing addendum",)
    assert len(ledger.annulment_lines(role_hash)) == 2


def test_ledger_refuses_annulment_after_judged_content_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep a revealed calibration look spent under RIDER-2's scope restriction."""

    ledger = AccessLedger(tmp_path / "ACCESS_LEDGER")
    role_hash = bank_module._FROZEN_A15_ROLE_HASH
    ledger.reserve_look(
        role_hash,
        "within-family-calibration",
        "post-M1",
        ("judged-row",),
        "capacity unlock",
        informative_judgments=100,
    )
    monkeypatch.setattr(
        access_module,
        "_LICENSED_ANNULMENTS",
        frozenset({(29, "within-family-calibration", 1)}),
    )

    assert access_module._ANNULMENT_SCOPE_BASES == frozenset({"synthetic_only_manifest"})
    assert not ledger._scope_allows_annulment(  # noqa: SLF001
        {"state": "RELEASED", "synthetic_only": True},
        "synthetic_only_manifest",
    )

    with pytest.raises(ValueError, match="judged content was unreleased"):
        ledger.annul_reservation(
            role_hash,
            "within-family-calibration",
            1,
            "The revealed look cannot regain its budget.",
            authority_addendum=29,
            scope_basis="synthetic_only_manifest",
        )
    assert ledger.budget_usage(role_hash)["within-family-calibration"] == 1


def test_annulment_metadata_accepts_sentence_punctuation_and_requires_iso_date() -> None:
    """Validate ANNUL prose and dates without banning interior punctuation."""

    assert access_module._valid_annulment_reason("The fitted tau 0.5 was wrong.")
    assert access_module._valid_annulment_reason("The run used e.g. a bad root.")
    assert not access_module._valid_annulment_reason("No terminal punctuation")
    assert not access_module._valid_annulment_reason("One defect. Another defect.")
    assert access_module._valid_annulment_date("2026-08-21")
    assert not access_module._valid_annulment_date("x")
    assert not access_module._valid_annulment_date("2026-8-21")


def test_ledger_audit_discloses_noncalendar_annulment_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disclose an otherwise licensed ANNUL line carrying a non-date token."""

    ledger = AccessLedger(tmp_path / "ACCESS_LEDGER")
    role_hash = bank_module._FROZEN_A15_ROLE_HASH
    ledger.reserve_once(
        role_hash,
        access_module.H_JND_LEDGER_KEY,
        ("synthetic-row",),
        "synthetic branch probe",
        synthetic_only=True,
    )
    monkeypatch.setattr(
        access_module,
        "_LICENSED_ANNULMENTS",
        frozenset({(29, access_module.H_JND_LEDGER_KEY, 1)}),
    )
    ledger_path = tmp_path / "ACCESS_LEDGER" / f"{role_hash}.jsonl"
    target_line = ledger_path.read_bytes().splitlines(keepends=True)[0]
    invalid_date_annulment = {
        "state": "ANNUL",
        "role_hash": role_hash,
        "ledger_key": access_module.H_JND_LEDGER_KEY,
        "slot_index": 1,
        "annulled_line_sha256": hashlib.sha256(target_line).hexdigest(),
        "reason": "The synthetic spend was defective.",
        "authority_addendum": 29,
        "date": "x",
        "scope_basis": "synthetic_only_manifest",
    }
    with ledger_path.open("a", encoding="utf-8") as handle:
        encoded = json.dumps(invalid_date_annulment, sort_keys=True, separators=(",", ":"))
        handle.write(encoded + "\n")

    assert ledger.ledger_defects(role_hash) == ("line 2: ANNUL date is not an ISO calendar date",)


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
        load_bank((pilot,), (schedule,), family, schedule)
    with pytest.raises(PermissionError, match="quarantined"):
        load_bank((sealed,), (schedule,), family, schedule)
    with pytest.raises(PermissionError, match="recursive quarantined"):
        load_bank((bank_root,), (schedule,), family, schedule)


def test_frozen_recorded_bank_fixture_has_stable_loader_digest(tmp_path: Path) -> None:
    """A recorded real main-bank slice pins loader reconciliation output."""

    fixture_path = Path(__file__).parents[2] / "fixtures/ruler_v4_real_bank_slice.py"
    fixture = runpy.run_path(str(fixture_path))["FIXTURE"]
    bank_path = tmp_path / "main-s0001.jsonl"
    schedule_path = tmp_path / "session_manifest.jsonl"
    frozen_schedule_path = tmp_path / "PRESENTATION_SCHEDULE.jsonl"
    family_path = tmp_path / "A15_FAMILY_MAP.json"
    bank_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in fixture["bank_rows"]), encoding="utf-8"
    )
    schedule_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in fixture["manifest_rows"]),
        encoding="utf-8",
    )
    frozen_schedule_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in fixture["frozen_schedule_rows"]),
        encoding="utf-8",
    )
    graphs = fixture["graphs"]
    role_lines = "\n".join(
        f"{graph_hash}\t{graph['role']}" for graph_hash, graph in sorted(graphs.items())
    )
    family_path.write_text(
        json.dumps(
            {
                "graphs": graphs,
                "role_hash": hashlib.sha256(role_lines.encode("utf-8")).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    bank = _load_synthetic_bank(
        (bank_path,),
        (schedule_path,),
        family_path,
        frozen_schedule_path,
    )
    payload = {
        "source": fixture["source"],
        "report": asdict(bank.report),
        "role_hash": bank.role_hash,
        "rows": [
            (
                row.presentation_id,
                row.session_id,
                row.graph_hash,
                row.role,
                row.purpose.value,
                row.verdict,
                row.tie,
                row.confidence,
            )
            for row in bank.rows
        ],
        "guarded_test_count": bank.guarded_test_count,
        "expected_test_base_pairs": {
            role: list(base_pairs) for role, base_pairs in bank.expected_test_base_pairs.items()
        },
        "expected_test_graphs": {
            role: list(graph_hashes) for role, graph_hashes in bank.expected_test_graphs.items()
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    assert hashlib.sha256(encoded).hexdigest() == (
        "24a34d434024f21622bdd5d2fe323d933207781093ab0bdc0b99847282516a78"  # noqa: E501  # pragma: allowlist secret
    )


def test_bank_loader_validates_a13_labels_and_schedule_schema(tmp_path: Path) -> None:
    """Implicit abstains and contradictory labels fail closed at ingestion."""

    _, bank_path, schedule_path, family_path = _loaded_holdout_fixture(tmp_path)
    rows = [json.loads(line) for line in bank_path.read_text(encoding="utf-8").splitlines()]
    rows[0].update({"verdict": 0, "tie": False})
    bank_path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")
    implicit_abstain = _load_synthetic_bank(
        (bank_path,), (schedule_path,), family_path, schedule_path
    )
    assert implicit_abstain.report.excluded_invalid == 1

    rows[0].update({"verdict": 2, "tie": True})
    bank_path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="verdict and tie"):
        _load_synthetic_bank((bank_path,), (schedule_path,), family_path, schedule_path)

    schedule_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks presentation identities"):
        _load_synthetic_bank(
            (bank_path,),
            (schedule_path,),
            family_path,
            schedule_path,
        )


def test_weight_fit_refuses_nonfit_but_consumes_train_replication_rows() -> None:
    """Weights consume both train lines while JND receives only replication."""

    plan = FittingPlan(_weight_parameters())
    row = _synthetic_recovery_rows(count=1)[0]
    config = OptimizerConfig(steps=1)
    with pytest.raises(ValueError, match="non-train"):
        PairwiseObjective((replace(row, purpose=SplitPurpose.VALIDATE),), plan)
    original = replace(
        row,
        base_pair_id="replicated-pair",
        replicate_group_id="replicated-pair",
        session_id="session-a",
        blind_id_a="drawing-a",
        blind_id_b="drawing-b",
    )
    replication = replace(
        original,
        is_replication=True,
        session_id="session-b",
        blind_id_a="drawing-b",
        blind_id_b="drawing-a",
    )
    result = fit_weights(PairwiseObjective((replication,), plan), config)
    lines = partition_fit_ord_lines((original, replication))

    assert result.steps_completed == 1
    assert lines.train == (original, replication)
    assert lines.replication == (original, replication)
    with pytest.raises(ValueError, match="25-pair minimum"):
        fit_jnd_heterogeneity(
            lines.replication,
            plan,
            result.weights,
            JNDFitConfig(
                role_hash=bank_module._FROZEN_A15_ROLE_HASH,
                top_composite_pair_counts={"synthetic": 67},
                rotation_envelopes={"synthetic": 0.01},
            ),
        )
    real_result = fit_weights(
        PairwiseObjective((replace(replication, synthetic=False),), plan),
        config,
    )
    assert real_result.steps_completed == 1
    with pytest.raises(ValueError, match="non-train"):
        partition_fit_ord_lines((replace(original, purpose=SplitPurpose.VALIDATE), replication))


def test_objective_fences_observation_profiles() -> None:
    """One likelihood cannot mix score rows from different observation profiles."""

    row = _synthetic_recovery_rows(count=1)[0]
    with pytest.raises(ValueError, match="profile"):
        PairwiseObjective(
            (row, replace(row, observation_profile="other-profile")),
            FittingPlan(_weight_parameters()),
        )


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
            JNDFitConfig(
                role_hash=bank_module._FROZEN_A15_ROLE_HASH,
                top_composite_pair_counts={"synthetic": 67},
                rotation_envelopes={"synthetic": 0.01},
            ),
        )


def test_jnd_heterogeneity_requires_actual_cross_session_repeats() -> None:
    """Replication flags cannot substitute for cross-session repeated pairs."""

    plan = FittingPlan(_weight_parameters())
    source = _synthetic_recovery_rows(count=2)
    unswapped = tuple(
        replace(
            row,
            is_replication=True,
            base_pair_id=f"unique-{index}",
            replicate_group_id=f"unique-{index}",
            session_id=f"session-{index}",
        )
        for index, row in enumerate(source)
    )
    with pytest.raises(ValueError, match="cross-session replication"):
        fit_jnd_heterogeneity(
            unswapped,
            plan,
            {"w_structure": 0.6, "w_neighborhood": 1.6},
            JNDFitConfig(
                role_hash=bank_module._FROZEN_A15_ROLE_HASH,
                top_composite_pair_counts={"synthetic": 67},
                rotation_envelopes={"synthetic": 0.01},
            ),
        )


def test_repl_swap_qualifies_same_displayed_order_cross_session() -> None:
    """W-13 qualification does not invent a two-displayed-orders requirement."""

    source = _synthetic_recovery_rows(count=1)[0]
    first = replace(
        source,
        is_replication=True,
        replicate_group_id="same-order-group",
        base_pair_id="same-order-pair",
        session_id="session-a",
        blind_id_a="drawing-a",
        blind_id_b="drawing-b",
    )
    second = replace(first, session_id="session-b")

    counts = uncertainty_module._validate_replication_rows((first, second))

    assert counts == {(first.primary_class, first.size_band): 1}


def test_side_swap_controls_are_audit_only_and_join_by_group(tmp_path: Path) -> None:
    """Exchanged controls survive only in the typed FIT-ORD(b) audit channel."""

    _, bank_path, schedule_path, family_path = _loaded_holdout_fixture(tmp_path)
    schedule_rows = [
        json.loads(line) for line in schedule_path.read_text(encoding="utf-8").splitlines()
    ]
    bank_rows = [json.loads(line) for line in bank_path.read_text(encoding="utf-8").splitlines()]
    schedule_rows.append(
        {
            "presentation_id": "presentation-swap",
            "session_id": "session-swap",
            "base_pair_id": "pair-fit",
            "replicate_group_id": "pair-fit",
            "graph_hash": "fit-graph",
            "blind_id_A": "B-fit",
            "blind_id_B": "A-fit",
            "profile_opaque_id": "profile",
            "budget_line": "CONTROLS",
            "control_type": "side-swap-repeat",
            "partition": "train",
        }
    )
    bank_rows.append(
        {
            "presentation_id": "presentation-swap",
            "session_id": "session-swap",
            "base_pair_id": "pair-fit",
            "replicate_group_id": "pair-fit",
            "graph_hash": "fit-graph",
            "session_accepted": True,
            "instrument_hash": "instrument",
            "judge_id": "judge/CF@4",
            "verdict": 2,
            "tie": False,
            "confidence": 2,
            "side_bit": 1,
            "budget_line": "CONTROLS",
            "control_type": "side-swap-repeat",
        }
    )
    schedule_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in schedule_rows), encoding="utf-8"
    )
    bank_path.write_text("".join(f"{json.dumps(row)}\n" for row in bank_rows), encoding="utf-8")

    bank = _load_synthetic_bank((bank_path,), (schedule_path,), family_path, schedule_path)

    assert bank.report.side_swap_audit_rows == 1
    assert len(bank.side_swap_audit_rows) == 1
    assert isinstance(bank.side_swap_audit_rows[0], SideSwapAuditRow)
    assert all(row.session_id != "session-swap" for row in bank.rows)
    source = _synthetic_recovery_rows(count=1)[0]
    base = replace(
        source,
        replicate_group_id="pair-fit",
        base_pair_id="pair-fit",
        session_id="session-fit",
        blind_id_a="A-fit",
        blind_id_b="B-fit",
        graded_verdict=-2,
        outcome=-1,
    )
    repeat = replace(base, session_id="session-repeat")
    result = side_swap_audit((base, repeat), bank.side_swap_audit_rows)

    assert result.control_legs == result.resolved_legs == result.exact_reversals == 1
    assert result.order_effect == pytest.approx(0.0)
    assert result.likelihood_row_count == 0


def test_blind_attestation_digest_input_and_whitelist_are_derived(tmp_path: Path) -> None:
    """BLIND-ATTEST derives its gate from exact line bytes and rejects leaked keys."""

    fit_digest = "a" * 64
    map_digest = "b" * 64
    assertions = {
        "A1_QUARANTINE_DISJOINT": {
            "assertion": "A1_QUARANTINE_DISJOINT",
            "result": True,
            "evidence": {
                "quarantine_root": "p3/quarantine/blind-map",
                "fit_input_path_count": 1,
                "intersection_count": 0,
            },
        },
        "A2_SCHEMA_BLIND": {
            "assertion": "A2_SCHEMA_BLIND",
            "result": True,
            "evidence": {
                "fitpair_field_count": 27,
                "fitpair_field_list_sha256": "c" * 64,
                "engine_identity_fields": [],
            },
        },
        "A3_CODE_BLIND": {
            "assertion": "A3_CODE_BLIND",
            "result": True,
            "evidence": {
                "module_root": "dagua/eval/ruler_v4/fit",
                "files_scanned": 10,
                "matches": 0,
                "scanner_sha256": "d" * 64,
            },
        },
        "A4_RESOLUTION_COMPLETE": {
            "assertion": "A4_RESOLUTION_COMPLETE",
            "result": True,
            "evidence": {
                "base_pairs": 2,
                "resolved": 2,
                "distinct_pair_count": 2,
                "missing_blind_ids": 0,
                "duplicate_blind_ids": 0,
                "graph_hash_mismatches": 0,
            },
        },
    }
    payload = {
        "attestation_version": "v4-blind-attest-1",
        "date": "2026-08-22",
        "attester": "P5ACTIVATE",
        "map_path": "p3/quarantine/blind-map/map.jsonl",
        "map_sha256": map_digest,
        "map_rows": 2,
        "map_authority": "ADDENDUM-19",
        "fit_input_digest": fit_digest,
        "assertions": assertions,
    }
    line = f"{json.dumps(payload, sort_keys=True, separators=(',', ':'))}\n".encode()
    path = tmp_path / "attest.jsonl"
    path.write_bytes(line)
    line_digest = hashlib.sha256(line).hexdigest()

    verified = driver_module._verify_blind_attestation(
        path,
        line_digest,
        map_digest,
        fit_digest,
    )

    assert verified["fit_input_digest"] == fit_digest
    with pytest.raises(FitStartConditionError, match="input digest"):
        driver_module._verify_blind_attestation(path, line_digest, map_digest, "e" * 64)
    leaked = dict(payload)
    leaked["graph_name"] = "forbidden"
    leaked_line = f"{json.dumps(leaked, sort_keys=True, separators=(',', ':'))}\n".encode()
    path.write_bytes(leaked_line)
    with pytest.raises(FitStartConditionError, match="schema fields"):
        driver_module._verify_blind_attestation(
            path,
            hashlib.sha256(leaked_line).hexdigest(),
            map_digest,
            fit_digest,
        )


def test_dof_declaration_gate_is_complete_external_and_realized(tmp_path: Path) -> None:
    """DOF-DECL verifies its digest, completeness, buckets, and realized identities."""

    universal = [f"u-{index}" for index in range(9)]
    semantic = [f"s-{index}" for index in range(3)]
    group = ["mu", "tau_class", "tau_band", "lapse_rate"]
    aggregation = ["blend_mixing_weight", "cvar_alpha", "smoothed_max_temperature"]
    buckets = [
        {"bucket": "N_u", "cap": 9, "assignable": True, "filled": universal},
        {"bucket": "UNSPENT", "cap": 1, "assignable": False, "filled": []},
        {"bucket": "N_s", "cap": 3, "assignable": True, "filled": semantic},
        {"bucket": "N_g", "cap": 4, "assignable": True, "filled": group},
        {"bucket": "N_t", "cap": 3, "assignable": True, "filled": aggregation},
    ]
    source_digest = "f" * 64
    declaration = {
        "schema_version": "v4-dof-decl-1",
        "source_allocation_sha256": source_digest,
        "assignment_complete": True,
        "assignment_authority": "ADDENDUM-30",
        "buckets": buckets,
        "sum": 20,
        "allowed_fitted_dof": 20,
        "controlled_stimulus_fitted_dof": 0,
        "prior_floor_facets": ["U12", "U13", "U34"],
    }
    declaration_path = tmp_path / "FITTED_DOF_DECLARATION.json"
    declaration_path.write_text(json.dumps(declaration), encoding="utf-8")
    expected_digest = hashlib.sha256(declaration_path.read_bytes()).hexdigest()
    bucket_by_identity = {
        **{identity: "universal" for identity in universal},
        **{identity: "semantic" for identity in semantic},
        **{identity: "group_model" for identity in group},
        **{identity: "aggregation" for identity in aggregation},
    }
    weight_table = SimpleNamespace(
        entries=tuple(
            SimpleNamespace(fitted_parameter=identity) for identity in universal + semantic
        ),
        fitted_parameter_buckets=bucket_by_identity,
        dof_account=SimpleNamespace(within_cap=True, within_buckets=True, used=19),
        d_power=20,
        prior_floors={"U12": 0.5, "U13": 0.5, "U34": 0.5},
    )
    plan = SimpleNamespace(
        weights=tuple(SimpleNamespace(name=identity) for identity in universal + semantic),
        prior_floors=weight_table.prior_floors,
    )
    with patch.object(driver_module, "_allocation_block_sha256", return_value=source_digest):
        verified = driver_module._verify_dof_declaration(
            declaration_path,
            expected_digest,
            tmp_path / "PREREG.md",
            plan,
            weight_table,
        )

    assert verified["assignment_complete"] is True
    declaration["assignment_complete"] = False
    declaration_path.write_text(json.dumps(declaration), encoding="utf-8")
    with (
        patch.object(driver_module, "_allocation_block_sha256", return_value=source_digest),
        pytest.raises(FitStartConditionError, match="incomplete"),
    ):
        driver_module._verify_dof_declaration(
            declaration_path,
            hashlib.sha256(declaration_path.read_bytes()).hexdigest(),
            tmp_path / "PREREG.md",
            plan,
            weight_table,
        )


def test_jnd_replication_accepts_realized_same_order_side_bits(tmp_path: Path) -> None:
    """W-13 accepts realized repeat legs whose schedule keeps one displayed order."""

    _, bank_path, schedule_path, family_path = _loaded_holdout_fixture(tmp_path)
    repeat_schedule = {
        "presentation_id": "presentation-fit-repeat",
        "session_id": "session-fit-repeat",
        "base_pair_id": "pair-fit",
        "replicate_group_id": "pair-fit",
        "graph_hash": "fit-graph",
        "blind_id_A": "A-fit",
        "blind_id_B": "B-fit",
        "profile_opaque_id": "profile",
        "budget_line": "PRIMARY",
        "partition": "train",
    }
    repeat_bank = {
        "presentation_id": "presentation-fit-repeat",
        "session_id": "session-fit-repeat",
        "base_pair_id": "pair-fit",
        "replicate_group_id": "pair-fit",
        "graph_hash": "fit-graph",
        "session_accepted": True,
        "instrument_hash": "instrument",
        "judge_id": "judge/CF@4",
        "verdict": 1,
        "tie": False,
        "confidence": 2,
        "side_bit": 1,
    }
    with schedule_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(repeat_schedule)}\n")
    with bank_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(repeat_bank)}\n")

    bank = _load_synthetic_bank((bank_path,), (schedule_path,), family_path, schedule_path)
    assert len(bank.rows) == 2
    assert all(row.is_replication for row in bank.rows)
    assert {(row.blind_id_a, row.blind_id_b) for row in bank.rows} == {("A-fit", "B-fit")}

    source = _synthetic_recovery_rows(count=2)
    repetitions = tuple(
        replace(
            source[index],
            is_replication=row.is_replication,
            replicate_group_id=row.replicate_group_id,
            base_pair_id=row.base_pair_id,
            session_id=row.session_id,
            blind_id_a=row.blind_id_a,
            blind_id_b=row.blind_id_b,
            graph_hash=row.graph_hash,
            primary_class=row.primary_class,
            size_band=row.size_band,
            generator_family=row.generator_family,
        )
        for index, row in enumerate(bank.rows)
    )
    with pytest.raises(ValueError, match="25-pair minimum"):
        fit_jnd_heterogeneity(
            repetitions,
            FittingPlan(_weight_parameters()),
            {"w_structure": 0.6, "w_neighborhood": 1.6},
            JNDFitConfig(
                role_hash=bank_module._FROZEN_A15_ROLE_HASH,
                top_composite_pair_counts={"band": 67},
                rotation_envelopes={"class": 0.01},
            ),
        )


def test_c06_audits_free_coordinates_without_shrinking_heterogeneity() -> None:
    """C-06 compares parameter count, while preserving reported smoother traces."""

    class_effects = {"a": -0.6, "b": -0.2, "c": 0.25, "d": 0.7}
    band_effects = {"x": -0.25, "y": 0.0, "z": 0.3}
    observations = {
        (primary_class, size_band): uncertainty_module._CellObservation(
            estimate=class_effect + band_effect,
            variance=0.1,
        )
        for primary_class, class_effect in class_effects.items()
        for size_band, band_effect in band_effects.items()
    }
    initial = uncertainty_module._meta_fit(observations, frozenset(band_effects))

    fitted, actions, partial_declaration = uncertainty_module._apply_c06_shrink(
        observations,
        frozenset(band_effects),
        initial,
        frozenset(),
    )

    assert initial.effective_dof_class + initial.effective_dof_band > 2.0
    assert uncertainty_module._c06_parameter_count(initial) == 2
    assert actions == ()
    assert not partial_declaration
    assert fitted.tau_class > 0.0
    assert fitted.tau_band > 0.0
    assert fitted is initial
    assert len(set(fitted.cell_logs.values())) > 1


def test_c06_heterogeneous_block_does_not_collapse_to_pooled() -> None:
    """Preserve a heterogeneous fit under the rider-frozen C-06 comparand."""

    class_effects = {f"class-{index}": (index - 4.5) * 0.18 for index in range(10)}
    band_effects = {f"band-{index}": (index - 2.5) * 0.12 for index in range(6)}
    observations = {
        (primary_class, size_band): uncertainty_module._CellObservation(
            estimate=class_effect + band_effect,
            variance=0.02,
        )
        for primary_class, class_effect in class_effects.items()
        for size_band, band_effect in band_effects.items()
    }
    initial = uncertainty_module._meta_fit(observations, frozenset(band_effects))

    fitted, actions, partial_declaration = uncertainty_module._apply_c06_shrink(
        observations,
        frozenset(band_effects),
        initial,
        frozenset(),
    )

    assert actions == ()
    assert not partial_declaration
    assert uncertainty_module._c06_parameter_count(fitted) == 2
    assert len(set(fitted.cell_logs.values())) > 1


def test_c06_shrinks_only_the_largest_trace_unnamed_drift_coordinate() -> None:
    """Freeze an unnamed C-06 offender without sacrificing named heterogeneity."""

    initial = SimpleNamespace(
        parameter_vector=np.zeros(6),
        active_components=(
            "tau_class",
            "tau_band",
            "tau_generator",
            "tau_extra",
            "tau_alpha",
        ),
        tau_class=0.5,
        tau_band=0.3,
        tau_generator=0.1,
        tau_extra=0.1,
        tau_alpha=0.1,
        effective_dof_class=3.6,
        effective_dof_band=2.1,
        effective_dof_generator=0.2,
        effective_dof_extra=0.9,
        effective_dof_alpha=0.2,
    )
    after_extra = SimpleNamespace(
        **vars(initial),
    )
    after_extra.parameter_vector = np.zeros(5)
    after_extra.active_components = ("tau_class", "tau_band", "tau_generator", "tau_alpha")
    after_extra.tau_extra = 0.0
    after_alpha = SimpleNamespace(**vars(after_extra))
    after_alpha.parameter_vector = np.zeros(4)
    after_alpha.active_components = ("tau_class", "tau_band", "tau_generator")
    after_alpha.tau_alpha = 0.0
    refitted = SimpleNamespace(
        **vars(after_alpha),
    )
    refitted.parameter_vector = np.zeros(3)
    refitted.active_components = ("tau_class", "tau_band")
    refitted.tau_generator = 0.0

    with patch.object(
        uncertainty_module,
        "_meta_fit",
        side_effect=(after_extra, after_alpha, refitted),
    ) as meta_fit:
        fitted, actions, partial_declaration = uncertainty_module._apply_c06_shrink(
            {},
            frozenset(),
            initial,
            frozenset(),
        )

    assert fitted is refitted
    assert actions == ("tau_extra", "tau_alpha", "tau_generator")
    assert not partial_declaration
    assert fitted.tau_class == initial.tau_class
    assert fitted.tau_band == initial.tau_band
    assert [call.args[2] for call in meta_fit.call_args_list] == [
        frozenset({"tau_extra"}),
        frozenset({"tau_alpha", "tau_extra"}),
        frozenset({"tau_alpha", "tau_extra", "tau_generator"}),
    ]


def test_c06_point_effects_drift_publishes_partial_declaration() -> None:
    """Declare PARTIAL when point effects cannot be frozen at a named null prior."""

    point_effects = tuple(
        [f"u_class-{index}" for index in range(4)] + [f"v_band-{index}" for index in range(3)]
    )
    initial = SimpleNamespace(
        parameter_vector=np.zeros(len(point_effects) + 1),
        active_components=point_effects,
        **{component: 0.1 for component in point_effects},
    )

    with patch.object(uncertainty_module, "_meta_fit", return_value=initial):
        fitted, actions, partial_declaration = uncertainty_module._apply_c06_shrink(
            {},
            frozenset(),
            initial,
            frozenset(),
        )

    assert fitted is initial
    assert actions == ()
    assert partial_declaration
    assert uncertainty_module._c06_parameter_count(fitted) == 7


def test_c06_boundary_disclosure_covers_unnamed_block_component() -> None:
    """Publish an upper-bound disclosure for every active C-06 block component."""

    fitted = SimpleNamespace(
        active_components=("tau_class", "tau_generator"),
        tau_class=0.5,
        tau_generator=10.0,
        effective_dof_class=3.6,
        effective_dof_generator=0.4,
    )

    disclosures = uncertainty_module._variance_boundary_disclosures(fitted)

    assert len(disclosures) == 1
    assert disclosures[0].component == "tau_generator"
    assert disclosures[0].fitted_value == 10.0
    assert disclosures[0].effective_dof == 0.4


def test_w13_estimator_publishes_uncertainty_guards_and_one_shot_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W-13 returns every mandatory publication and ledgers its branch once."""

    monkeypatch.setattr(access_module, "_ACCESS_LEDGER_ROOT", tmp_path / "ACCESS_LEDGER")
    supported_rows = _jnd_success_rows()
    sparse_rows = tuple(
        replace(
            row,
            size_band="band-sparse",
            graph_hash="graph-sparse",
            base_pair_id="pair-sparse",
            replicate_group_id="pair-sparse",
        )
        for row in supported_rows[:2]
    )
    rows = supported_rows + sparse_rows
    plan = FittingPlan(_weight_parameters())
    config = JNDFitConfig(
        role_hash=bank_module._FROZEN_A15_ROLE_HASH,
        top_composite_pair_counts={"band-1": 67, "band-2": 67, "band-sparse": 0},
        rotation_envelopes={"class-1": 0.001, "class-2": 0.001},
    )
    with pytest.raises(ValueError, match="verified assignment"):
        fit_jnd_heterogeneity(
            tuple(replace(row, synthetic=False) for row in rows),
            plan,
            {"w_structure": 0.6, "w_neighborhood": 1.6},
            config,
        )
    with patch.object(
        uncertainty_module,
        "_meta_fit",
        wraps=uncertainty_module._meta_fit,
    ) as meta_fit:
        fit = fit_jnd_heterogeneity(
            rows,
            plan,
            {"w_structure": 0.6, "w_neighborhood": 1.6},
            config,
        )

    assert config.minimum_cell_count == 25
    assert config.q_band == 67
    assert config.bootstrap_replicates == 2000
    assert set(fit.cell_counts.values()) == {1, 50}
    assert set(fit.cell_jnd) == set(fit.cell_jnd_ci)
    assert set(fit.jnd_by_cell) == set(fit.cell_counts)
    assert ("class-1", "band-sparse") in fit.unestimated_cells
    assert fit.jnd_by_cell[("class-1", "band-sparse")] == pytest.approx(math.exp(fit.mu))
    assert fit.tau_class_ci.log_scale[0] <= fit.tau_class_ci.log_scale[1]
    assert fit.tau_band_ci.ratio_scale[0] <= fit.tau_band_ci.ratio_scale[1]
    assert fit.spread_ci_graph_clusters[0] <= fit.spread_ci_graph_clusters[1]
    assert fit.spread_ci_generator_families[0] <= fit.spread_ci_generator_families[1]
    assert 0.0 < fit.bootstrap_drop_rate.graph_clusters <= 1.0
    assert 0.0 <= fit.bootstrap_drop_rate.generator_families <= 1.0
    assert fit.effective_dof >= 0.0
    assert fit.n_jnd <= 2
    assert set(fit.c06_component_audit) == {"tau_class", "tau_band"}
    assert fit.c06_component_audit["tau_class"].realized_levels == 2
    assert fit.c06_component_audit["tau_band"].realized_levels == 2
    assert all(audit.effective_dof >= 0.0 for audit in fit.c06_component_audit.values())
    assert fit.loss_path
    assert meta_fit.call_count > config.bootstrap_replicates
    assert set(fit.c06_shrink_actions).isdisjoint(fit.split_half.frozen_components)
    assert not fit.uncalibrated_classes
    assert set(fit.tie_rates_by_class) == {"class-1", "class-2"}
    with pytest.raises(TypeError, match="minimum_cell_count"):
        JNDFitConfig(
            role_hash=bank_module._FROZEN_A15_ROLE_HASH,
            top_composite_pair_counts={"band-1": 67},
            rotation_envelopes={"class-1": 0.001},
            minimum_cell_count=20,
        )

    branch_fit = replace(
        fit,
        pooled_jnd_ci=(1.0, 1.1),
        spread_ci_graph_clusters=(1.2, 1.4),
    )
    assert not fit.c06_partial_declaration
    with pytest.raises(TypeError, match="ledger"):
        evaluate_h_jnd_branch(branch_fit)  # type: ignore[call-arg]
    branch_ledger = AccessLedger()
    branch = evaluate_h_jnd_branch(branch_fit, branch_ledger)
    assert branch.shipped_band == "class-conditional"
    assert branch.spread_lower == 1.2
    assert branch.pooled_ratio == pytest.approx(1.1)
    with pytest.raises(RuntimeError, match="once-only"):
        evaluate_h_jnd_branch(branch_fit, branch_ledger)


def test_v4_half_1_reproduces_frozen_table_anchor() -> None:
    """The landed A15 family map reproduces the frozen 102-unit partition."""

    family_map = (
        Path.home()
        / ".claude"
        / "research"
        / "dagua"
        / "ruler_v4"
        / "p3"
        / "frozen"
        / "A15_FAMILY_MAP.json"
    )
    if not family_map.exists():
        pytest.skip("campaign A15 family map is not installed")

    assignment = load_half_assignment(family_map)

    expected_digest = "".join(
        (
            "4041736f049333ca031409e8201b9834",  # pragma: allowlist secret
            "3b28afb93dd521382762f8072fd69b88",  # pragma: allowlist secret
        )
    )
    assert assignment.table_sha256 == expected_digest
    assert len(assignment.unit_halves) == 102
    assert len(assignment.graph_halves) == 116
    assert tuple(assignment.unit_halves.values()).count(0) == 52
    assert tuple(assignment.unit_halves.values()).count(1) == 50


def test_one_half_assignment_reaches_jnd_and_outer_weight_call_sites() -> None:
    """Both W-13 calls consume one shared partition also usable by weights."""

    rows = _jnd_success_rows()
    graphs = sorted({row.graph_hash for row in rows})
    assignment = HalfAssignment(
        graph_halves={graph: index % 2 for index, graph in enumerate(graphs)},
        unit_halves={f"unit-{index}": index % 2 for index in range(len(graphs))},
        table_sha256="0" * 64,
        family_map_sha256="1" * 64,
    )
    plan = FittingPlan(_weight_parameters())
    weights = {"w_structure": 0.6, "w_neighborhood": 1.6}
    config = JNDFitConfig(
        role_hash=bank_module._FROZEN_A15_ROLE_HASH,
        top_composite_pair_counts={"band-1": 67, "band-2": 67},
        rotation_envelopes={"class-1": 0.001, "class-2": 0.001},
    )
    with (
        patch.object(
            uncertainty_module,
            "_split_half_fit",
            wraps=uncertainty_module._split_half_fit,
        ) as split_fit,
        patch.object(
            uncertainty_module,
            "_bootstrap_spread",
            return_value=((1.0, 1.0), 0.0, (1.0, 1.0)),
        ),
    ):
        uncertainty_module.profile_jnd_block(
            rows,
            plan,
            weights,
            config,
            half_assignment=assignment,
        )
        fit_jnd_heterogeneity(
            rows,
            plan,
            weights,
            config,
            half_assignment=assignment,
        )

    assert split_fit.call_count == 2
    assert all(call.args[3] is assignment for call in split_fit.call_args_list)
    outer_halves = tuple(
        tuple(row for row in rows if assignment.graph_halves[row.graph_hash] == half)
        for half in (0, 1)
    )
    assert all(outer_halves)


def test_heterogeneous_w13_fit_reaches_class_conditional_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reach the H-JND class-conditional branch from fitted heterogeneity."""

    monkeypatch.setattr(access_module, "_ACCESS_LEDGER_ROOT", tmp_path / "ACCESS_LEDGER")
    cell_jnds = {
        ("class-1", "band-1"): 0.05,
        ("class-1", "band-2"): 0.25,
        ("class-2", "band-1"): 0.25,
        ("class-2", "band-2"): 1.25,
    }
    rows = _jnd_success_rows(cell_jnds)
    fit = fit_jnd_heterogeneity(
        rows,
        FittingPlan(_weight_parameters()),
        {"w_structure": 0.6, "w_neighborhood": 1.6},
        JNDFitConfig(
            role_hash=bank_module._FROZEN_A15_ROLE_HASH,
            top_composite_pair_counts={"band-1": 67, "band-2": 67},
            rotation_envelopes={"class-1": 0.001, "class-2": 0.001},
        ),
    )

    branch = evaluate_h_jnd_branch(fit, AccessLedger())

    assert fit.tau_class > 0.0
    assert fit.tau_band > 0.0
    assert fit.c06_shrink_actions == ()
    assert branch.shipped_band == "class-conditional"
    assert branch.spread_lower > branch.pooled_ratio


def test_freeze1_driver_fails_closed_for_holdouts_and_real_rows(tmp_path: Path) -> None:
    """Real activation opens only after all six gates and artifacts verify."""

    rows = _jnd_success_rows()
    plan = FittingPlan(_weight_parameters())
    config = JNDFitConfig(
        role_hash=bank_module._FROZEN_A15_ROLE_HASH,
        top_composite_pair_counts={"band-1": 67, "band-2": 67},
        rotation_envelopes={"class-1": 0.001, "class-2": 0.001},
    )
    with pytest.raises(ValueError, match="train-role"):
        run_freeze1_fit(
            (replace(rows[0], purpose=SplitPurpose.VALIDATE),),
            plan,
            config,
            tmp_path / "holdout-run",
            tmp_path / "holdout-ledger",
        )
    real_rows = tuple(replace(row, synthetic=False) for row in rows)
    with pytest.raises(FitStartConditionError, match="campaign completion"):
        run_freeze1_fit(
            real_rows,
            plan,
            config,
            tmp_path / "premature-real-run",
            tmp_path / "premature-real-ledger",
        )
    ready = RealFitStartConditions(
        campaign_complete=True,
        protocol_start_authorized=True,
        lapse_prior_frozen=True,
        graph_half_assignment_frozen=True,
        fitted_dof_declaration_verified=True,
        blind_map_attested=True,
    )
    driver_module._require_real_start_conditions(ready)
    for gate in asdict(ready):
        with pytest.raises(FitStartConditionError, match="every protocol start condition"):
            driver_module._require_real_start_conditions(replace(ready, **{gate: False}))
    with pytest.raises(FitStartConditionError, match="gate artifacts are missing"):
        run_freeze1_fit(
            real_rows,
            plan,
            config,
            tmp_path / "gated-real-run",
            tmp_path / "gated-real-ledger",
            real_start_conditions=ready,
        )
    with pytest.raises(TypeError, match="joint_tolerance"):
        FitDriverConfig(joint_tolerance=1.0e-6)  # type: ignore[call-arg]
    assert not (tmp_path / "holdout-run").exists()
    assert not (tmp_path / "premature-real-run").exists()
    assert not (tmp_path / "gated-real-run").exists()


def test_freeze1_synthetic_driver_cannot_construct_default_campaign_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Require the P5BR2 construction to inject a non-campaign ledger root."""

    source = inspect.getsource(run_freeze1_fit)
    assert "AccessLedger()" not in source
    campaign_root = tmp_path / "CAMPAIGN_ACCESS_LEDGER"
    monkeypatch.setattr(access_module, "_ACCESS_LEDGER_ROOT", campaign_root)
    with pytest.raises(ValueError, match="cannot target the campaign ledger root"):
        run_freeze1_fit(
            _jnd_success_rows(),
            FittingPlan(_weight_parameters()),
            JNDFitConfig(
                role_hash=bank_module._FROZEN_A15_ROLE_HASH,
                top_composite_pair_counts={"band-1": 67, "band-2": 67},
                rotation_envelopes={"class-1": 0.001, "class-2": 0.001},
            ),
            tmp_path / "blocked-run",
            campaign_root,
        )
    with pytest.raises(ValueError, match="cannot target the campaign ledger root"):
        run_freeze1_fit(
            _jnd_success_rows(),
            FittingPlan(_weight_parameters()),
            JNDFitConfig(
                role_hash=bank_module._FROZEN_A15_ROLE_HASH,
                top_composite_pair_counts={"band-1": 67, "band-2": 67},
                rotation_envelopes={"class-1": 0.001, "class-2": 0.001},
            ),
            tmp_path / "blocked-subtree-run",
            campaign_root / "synthetic-subtree",
        )
    assert not campaign_root.exists()
    assert not (tmp_path / "blocked-run").exists()
    assert not (tmp_path / "blocked-subtree-run").exists()


def test_freeze1_driver_runs_synthetic_fit_with_ledgered_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic FREEZE-1 runs end to end with fixed-point and budget evidence."""

    monkeypatch.setattr(access_module, "_ACCESS_LEDGER_ROOT", tmp_path / "ACCESS_LEDGER")
    rows = _jnd_success_rows()
    plan = FittingPlan(_weight_parameters())
    config = JNDFitConfig(
        role_hash=bank_module._FROZEN_A15_ROLE_HASH,
        top_composite_pair_counts={"band-1": 67, "band-2": 67},
        rotation_envelopes={"class-1": 0.001, "class-2": 0.001},
    )
    run_dir = tmp_path / "freeze1-run"
    synthetic_ledger_root = tmp_path / "SYNTHETIC_ACCESS_LEDGER"

    result = run_freeze1_fit(rows, plan, config, run_dir, synthetic_ledger_root)

    assert result.trajectory
    assert result.trajectory[0].joint_improvement is None
    assert result.trajectory[-1].joint_improvement is not None
    assert result.trajectory[-1].joint_improvement <= 1.0e-10
    assert all(
        math.isfinite(iteration.joint_objective)
        and iteration.weight_lapse_improvement >= 0.0
        and (iteration.jnd_improvement is None or iteration.jnd_improvement >= 0.0)
        and (iteration.joint_improvement is None or iteration.joint_improvement >= 0.0)
        for iteration in result.trajectory
    )
    assert 0.0 <= result.lapse_rate <= 0.25
    assert result.lapse_interval[0] <= result.lapse_rate <= result.lapse_interval[1]
    assert result.lapse_prior_weight == pytest.approx(111.0 / (111.0 + len(rows)))
    assert result.lapse_boundary_disclosure is None
    assert 1.0e-6 <= result.lapse_prior_free_sensitivity.lapse_rate <= 0.25
    assert result.access_budget_before["test-h-jnd-branch"] == 0
    assert result.access_budget_after["test-h-jnd-branch"] == 1
    assert not result.jnd_fit.uncalibrated_classes
    assert not result.jnd_fit.c06_partial_declaration
    assert {path.name for path in run_dir.iterdir()} == {
        "manifest.json",
        "result.json",
        "status.json",
        "trajectory.jsonl",
    }
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    publication = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    trajectory = [
        json.loads(line)
        for line in (run_dir / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["seed"] == 20260811
    assert manifest["joint_tolerance"] == 1.0e-10
    assert manifest["synthetic_only"] is True
    assert manifest["row_count"] == len(rows)
    assert manifest["replication_row_count"] == len(rows)
    assert publication["access_budget_after"]["test-h-jnd-branch"] == 1
    assert publication["ledger_annulments"] == []
    assert publication["ledger_defects"] == []
    assert publication["jnd"]["c06_partial_declaration"] is False
    assert publication["lapse_interval"] == list(result.lapse_interval)
    assert publication["lapse_prior_weight"] == pytest.approx(result.lapse_prior_weight)
    assert publication["lapse_boundary_disclosure"] is None
    assert publication["iterations"] == len(result.trajectory) == len(trajectory)
    assert status == {"state": "COMPLETE"}
    with pytest.raises(FileExistsError):
        run_freeze1_fit(rows, plan, config, run_dir, synthetic_ledger_root)
    role_hash = bank_module._FROZEN_A15_ROLE_HASH
    assert AccessLedger(synthetic_ledger_root).budget_usage(role_hash)["test-h-jnd-branch"] == 1
    assert AccessLedger().budget_usage(role_hash)["test-h-jnd-branch"] == 0
