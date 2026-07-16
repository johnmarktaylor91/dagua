"""Tests for the W5 native finisher wrapper."""

from __future__ import annotations

import time

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.native_finisher import W5Seed, run_w5_finisher


def test_w5_finisher_skips_when_no_budget() -> None:
    """The predicted-cost entry gate skips W5 when the deadline is exhausted."""
    config = LayoutConfig()
    config._dagua_native_deadline_s = time.perf_counter() - 1.0
    pos = torch.zeros((4, 2), dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 10.0)

    result = run_w5_finisher(
        seeds=[W5Seed("incumbent", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=lambda candidate: 0.0,
        current_best_score=0.0,
        is_semantically_directed=False,
        declared_hierarchical=False,
        config=config,
    )

    assert result.skipped_reason == "no_budget"
    assert result.candidates == ()
    assert result.checkpoints == ()


def test_w5_finisher_drops_nonfinite_seed() -> None:
    """Non-finite warm starts are ignored instead of scored."""
    pos = torch.full((4, 2), float("nan"), dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 10.0)

    result = run_w5_finisher(
        seeds=[W5Seed("bad", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=lambda candidate: 1.0,
        current_best_score=0.0,
        is_semantically_directed=False,
        declared_hierarchical=False,
    )

    assert result.skipped_reason == "no_finite_seed"
    assert result.candidates == ()


def test_w5_finisher_never_accepts_below_incumbent_score() -> None:
    """Rejected checkpoints cannot become returned W5 candidates."""
    pos = torch.tensor(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]],
        dtype=torch.float32,
    )
    edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 2.0)

    result = run_w5_finisher(
        seeds=[W5Seed("incumbent", pos)],
        edge_index=edge_index,
        node_sizes=node_sizes,
        score_fn=lambda candidate: -1.0,
        current_best_score=0.0,
        is_semantically_directed=False,
        declared_hierarchical=False,
    )

    assert result.candidates == ()
    assert all(not checkpoint.accepted for checkpoint in result.checkpoints)
