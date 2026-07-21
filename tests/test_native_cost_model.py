"""Tests for native deterministic modeled-cost helpers."""

from __future__ import annotations

import pytest
import torch

from dagua.layout.ops.pipelines import native_cost_model as cost_model
from dagua.layout.ops.pipelines.native_cost_model import (
    PROVENANCE_REF,
    apsp_volume,
    estimate_native_work_cost,
    fcose_force_volume,
    ruler_sample_volume,
    stress_pair_volume,
    w5_step_volume,
)


def test_cost_model_volume_helpers() -> None:
    """Volume helpers expose deterministic complexity forms.

    Returns
    -------
    None
        Assertions validate known arithmetic.
    """
    assert fcose_force_volume(num_nodes=10, num_edges=15, steps=4) == pytest.approx(100.0)
    assert stress_pair_volume(num_nodes=5, steps=3) == pytest.approx(30.0)
    assert stress_pair_volume(num_nodes=5, steps=3, sample_pairs=7) == pytest.approx(21.0)
    assert apsp_volume(num_nodes=4, num_edges=6) == pytest.approx(40.0)
    assert ruler_sample_volume(num_nodes=4, num_edges=6, samples=12) == pytest.approx(12.0)
    assert w5_step_volume(mode="barrier_2d", steps=8, seeds=2, checkpoints=3) == pytest.approx(22.0)


def test_estimate_native_work_cost_uses_problem_shape_and_w5_tiny_constants() -> None:
    """W5 estimator preserves the tiny-row calibrated constants.

    Returns
    -------
    None
        Assertions validate generation/referee costs and metadata.
    """
    problem = {
        "num_nodes": 4,
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
    }

    cost = estimate_native_work_cost(
        problem,
        family="w5",
        knobs={"steps": 96, "seeds": 1, "checkpoints": 2, "mode": "unit"},
        device_class="cpu",
    )

    assert cost.family == "w5"
    assert cost.generation_dwu == pytest.approx(96.0 * 0.0437)
    assert cost.reserved_score_dwu == pytest.approx(2.0 * 0.019)
    assert cost.metadata["num_nodes"] == 4
    assert cost.metadata["num_edges"] == 3
    assert cost.metadata["provenance"] == PROVENANCE_REF
    assert cost.metadata["terms"] == {"step": 96.0, "referee": 2.0, "combined": 98.0}


def test_estimate_native_work_cost_prices_unknown_family_as_opaque() -> None:
    """Unknown families receive a conservative opaque full-arm prior.

    Returns
    -------
    None
        Assertions validate opaque fallback behavior.
    """
    cost = estimate_native_work_cost(
        {"num_nodes": 10, "num_edges": 20},
        family="new_arm",
        knobs={"volume": 3},
        device_class="cpu",
    )

    assert cost.family == "new_arm"
    assert cost.generation_dwu == pytest.approx(30.0)
    assert cost.reserved_score_dwu == pytest.approx(0.0)
    assert cost.metadata["terms"] == {"full_arm": 3.0}


def test_directed_flat_arm_families_use_calibrated_priors() -> None:
    """Directed challenger arms price through frozen table-backed families.

    Returns
    -------
    None
        Assertions validate the calibrated flat packages used by directed
        ledger admission.
    """
    problem = {"num_nodes": 500, "num_edges": 1470}

    sugiyama = estimate_native_work_cost(
        problem,
        family="directed_sugiyama",
        knobs={"volume": 1},
        device_class="cpu",
    )
    recombinant = estimate_native_work_cost(
        problem,
        family="directed_recombinant",
        knobs={"volume": 1},
        device_class="cpu",
    )

    assert sugiyama.generation_dwu == pytest.approx(2.2)
    assert recombinant.generation_dwu == pytest.approx(2.5)
    assert sugiyama.metadata["provenance"] == PROVENANCE_REF


def test_fcose_exact_regime_prices_small_rows_at_true_tiny_cost() -> None:
    """Exact-repulsion fCoSE rows (N <= 512) price near their real ~0.2-0.6s cost.

    The M2 4-row regression class: the old single linear prior over-priced
    small/medium rows 90-7000x (sbm_low_mix predicted 2229.7 DWU vs 0.32s
    real), starving the winning fCoSE contest arms. C2 telemetry prices the
    exact regime at its true tiny cost so the arms admit.
    """
    sbm_low_mix_shape = {"num_nodes": 100, "num_edges": 642}

    cost = estimate_native_work_cost(
        sbm_low_mix_shape,
        family="fcose",
        knobs={"steps": 2500, "samples": None},
        device_class="cuda",
    )

    assert cost.metadata["fcose_regime"] == "exact"
    assert "force" in cost.metadata["terms"]
    assert cost.generation_dwu < 2.0
    # Admission headroom: 2.0 x (generation + score + 55s Arm-S prior) must fit
    # far under the ~280s remaining observed at the fCoSE seam on this row.
    assert 2.0 * (cost.generation_dwu + cost.reserved_score_dwu + 55.0) < 150.0


def test_fcose_barnes_hut_regime_keeps_scale_1k_priced_out() -> None:
    """Barnes-Hut fCoSE rows (N > 512) stay decisively unaffordable.

    r8_nested_scale_1k_budget real cost is ~6.5 s/step (~16,000s per full
    2500-step arm) on the calibration box; the frozen prior must keep it above
    2x the 300s benchmark budget with wide margin so the fCoSE-skip /
    Arm-S-admit scale anchor and the parity gate are preserved.
    """
    scale_1k_shape = {"num_nodes": 1000, "num_edges": 2038}

    cost = estimate_native_work_cost(
        scale_1k_shape,
        family="fcose",
        knobs={"steps": 2500, "samples": None},
        device_class="cuda",
    )

    assert cost.metadata["fcose_regime"] == "barnes_hut"
    assert "force_bh" in cost.metadata["terms"]
    assert cost.generation_dwu > 600.0

    cpu_cost = estimate_native_work_cost(
        scale_1k_shape,
        family="fcose",
        knobs={"steps": 2500, "samples": None},
        device_class="cpu",
    )

    assert cpu_cost.generation_dwu > 600.0


def test_fcose_regime_cliff_is_at_the_exact_repulsion_cap() -> None:
    """Pricing jumps by orders of magnitude exactly where the embedder regime flips."""
    knobs = {"steps": 2500, "samples": None}

    at_cap = estimate_native_work_cost(
        {"num_nodes": 512, "num_edges": 1024},
        family="fcose",
        knobs=knobs,
        device_class="cuda",
    )
    above_cap = estimate_native_work_cost(
        {"num_nodes": 513, "num_edges": 1024},
        family="fcose",
        knobs=knobs,
        device_class="cuda",
    )

    assert at_cap.metadata["fcose_regime"] == "exact"
    assert above_cap.metadata["fcose_regime"] == "barnes_hut"
    assert above_cap.generation_dwu > 100.0 * at_cap.generation_dwu


def test_fcose_barnes_hut_missing_term_falls_back_to_exact_pricing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A custom table without ``force_bh`` degrades to the exact-regime term.

    The fallback must never price a Barnes-Hut row at an accidental zero
    (free admission).
    """
    stripped = {
        key: {term: pair for term, pair in terms.items() if term != "force_bh"}
        for key, terms in cost_model.FROZEN_COST_TABLE.items()
    }
    monkeypatch.setattr(cost_model, "FROZEN_COST_TABLE", stripped)

    cost = cost_model.estimate_native_work_cost(
        {"num_nodes": 1000, "num_edges": 2038},
        family="fcose",
        knobs={"steps": 2500, "samples": None},
        device_class="cuda",
    )

    assert cost.metadata["fcose_regime"] == "barnes_hut"
    assert "force" in cost.metadata["terms"]
    assert cost.generation_dwu > 0.0
