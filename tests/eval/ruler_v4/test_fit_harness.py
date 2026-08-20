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
from typing import Dict, Optional
from unittest.mock import patch

import numpy as np
import pytest
import torch

import dagua.eval.ruler_v4.fit.access as access_module
import dagua.eval.ruler_v4.fit.bank as bank_module
from dagua.eval.ruler_v4.fit import (
    W08_LEDGER_KEY,
    AccessLedger,
    CalibrationLookConsumedError,
    CalibrationLookGuard,
    FitPair,
    FittingPlan,
    JNDFitConfig,
    JudgmentRow,
    OptimizerConfig,
    PairwiseObjective,
    ReusableJudgmentGuard,
    SceneRescorer,
    SplitPurpose,
    WeightParameter,
    apply_outer_weight_split_half,
    evaluate_h_jnd_branch,
    fit_jnd_heterogeneity,
    fit_pairs_from_rescoring,
    fit_weights,
    jnd_band_calibration,
    load_bank,
    ordered_response_calibration,
    partition_fit_ord_lines,
    partition_holdouts,
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


def _jnd_success_rows() -> tuple[FitPair, ...]:
    """Build four supported cells of true cross-session side swaps.

    Returns
    -------
    tuple[FitPair, ...]
        Two presentations for each of 25 base pairs in four cells.
    """

    source = _synthetic_recovery_rows(count=100)
    rows = []
    cells = (
        ("class-1", "band-1"),
        ("class-1", "band-2"),
        ("class-2", "band-1"),
        ("class-2", "band-2"),
    )
    for cell_index, cell in enumerate(cells):
        for pair_index in range(25):
            original = source[cell_index * 25 + pair_index]
            verdict = original.graded_verdict
            first = replace(
                original,
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
    assert first.weights["w_structure"] == pytest.approx(0.6431144, abs=1.0e-6)
    assert first.weights["w_neighborhood"] == pytest.approx(1.5997255, abs=1.0e-6)
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
    """A fixed prior contribution vanishes relative to growing evidence."""

    rows = _synthetic_recovery_rows(count=8)
    plan = FittingPlan(_weight_parameters())
    objective = PairwiseObjective(rows, plan)
    weights = torch.tensor((0.5, 2.0), dtype=torch.float64)
    penalty = objective.loss(weights) - objective.negative_log_likelihood(weights)
    expected_sum = sum((math.log(value) / math.log(4.0)) ** 2 for value in (0.5, 2.0))

    assert plan.prior_strength == 2.0
    assert float(penalty) == pytest.approx(2.0 * expected_sum / len(rows))
    with pytest.raises(TypeError, match="prior_strength"):
        FittingPlan(_weight_parameters(), prior_strength=0.1)


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
    with pytest.raises(ValueError, match="partial guarded role"):
        CalibrationLookGuard().consume(
            filtered,
            "within-family-calibration",
            "post-M1",
            100,
            "capacity unlock",
        )
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
        "ea01976a9ad1308de1f7205325a2d101dab1455b2e1bc2b35c64031e7d3a4046"  # noqa: E501  # pragma: allowlist secret
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
    with pytest.raises(NotImplementedError, match="lapse prior"):
        fit_weights(PairwiseObjective((replace(replication, synthetic=False),), plan), config)
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


def test_jnd_heterogeneity_requires_actual_cross_session_side_swaps() -> None:
    """Replication flags cannot substitute for repeated side-swapped pairs."""

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

    swapped = (
        replace(
            source[0],
            is_replication=True,
            base_pair_id="shared",
            replicate_group_id="shared",
            session_id="session-a",
            blind_id_a="drawing-a",
            blind_id_b="drawing-b",
        ),
        replace(
            source[1],
            is_replication=True,
            base_pair_id="shared",
            replicate_group_id="shared",
            session_id="session-b",
            blind_id_a="drawing-b",
            blind_id_b="drawing-a",
        ),
    )
    with pytest.raises(ValueError, match="25-pair minimum"):
        fit_jnd_heterogeneity(
            swapped,
            plan,
            {"w_structure": 0.6, "w_neighborhood": 1.6},
            JNDFitConfig(
                role_hash=bank_module._FROZEN_A15_ROLE_HASH,
                top_composite_pair_counts={"synthetic": 67},
                rotation_envelopes={"synthetic": 0.01},
            ),
        )


def test_w13_estimator_publishes_uncertainty_guards_and_one_shot_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W-13 returns every mandatory publication and ledgers its branch once."""

    monkeypatch.setattr(access_module, "_ACCESS_LEDGER_ROOT", tmp_path / "ACCESS_LEDGER")
    rows = _jnd_success_rows()
    plan = FittingPlan(_weight_parameters())
    config = JNDFitConfig(
        role_hash=bank_module._FROZEN_A15_ROLE_HASH,
        top_composite_pair_counts={"band-1": 67, "band-2": 67},
        rotation_envelopes={"class-1": 0.001, "class-2": 0.001},
    )
    with pytest.raises(NotImplementedError, match="graph-to-half"):
        fit_jnd_heterogeneity(
            tuple(replace(row, synthetic=False) for row in rows),
            plan,
            {"w_structure": 0.6, "w_neighborhood": 1.6},
            config,
        )
    fit = fit_jnd_heterogeneity(
        rows,
        plan,
        {"w_structure": 0.6, "w_neighborhood": 1.6},
        config,
    )

    assert config.minimum_cell_count == 25
    assert config.q_band == 67
    assert config.bootstrap_replicates == 2000
    assert set(fit.cell_counts.values()) == {25}
    assert set(fit.cell_jnd) == set(fit.cell_jnd_ci)
    assert fit.tau_class_ci.log_scale[0] <= fit.tau_class_ci.log_scale[1]
    assert fit.tau_band_ci.ratio_scale[0] <= fit.tau_band_ci.ratio_scale[1]
    assert fit.spread_ci_graph_clusters[0] <= fit.spread_ci_graph_clusters[1]
    assert fit.spread_ci_generator_families[0] <= fit.spread_ci_generator_families[1]
    assert 0.0 <= fit.bootstrap_drop_rate.graph_clusters <= 1.0
    assert 0.0 <= fit.bootstrap_drop_rate.generator_families <= 1.0
    assert fit.effective_dof >= 0.0
    assert fit.loss_path
    assert not fit.uncalibrated_classes
    assert set(fit.tie_rates_by_class) == {"class-1", "class-2"}
    with pytest.raises(TypeError, match="minimum_cell_count"):
        JNDFitConfig(
            role_hash=bank_module._FROZEN_A15_ROLE_HASH,
            top_composite_pair_counts={"band-1": 67},
            rotation_envelopes={"class-1": 0.001},
            minimum_cell_count=20,
        )

    branch = evaluate_h_jnd_branch(fit)
    r_pool = fit.pooled_jnd_ci[1] / fit.pooled_jnd_ci[0]
    expected = "class-conditional" if fit.spread_ci_graph_clusters[0] > r_pool else "pooled"
    assert branch.shipped_band == expected
    with pytest.raises(RuntimeError, match="once-only"):
        evaluate_h_jnd_branch(fit)
