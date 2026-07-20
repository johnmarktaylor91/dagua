"""Tests for native deterministic modeled-cost helpers."""

from __future__ import annotations

import pytest
import torch

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
    """W5 estimator preserves the tiny-row placeholder constants.

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
    assert cost.generation_dwu == pytest.approx(93.0)
    assert cost.reserved_score_dwu == pytest.approx(0.0)
    assert cost.metadata["terms"] == {"full_arm": 3.0}
