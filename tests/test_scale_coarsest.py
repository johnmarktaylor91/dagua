"""Tests for scale coarsest native wrapper behavior."""

from __future__ import annotations

import time
from typing import Tuple

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.state import LayoutProblem
from dagua.layout.scale import coarsest


def _chain_problem(num_nodes: int = 4) -> LayoutProblem:
    """Build a small deterministic chain problem.

    Parameters
    ----------
    num_nodes : int, default=4
        Number of nodes in the chain.

    Returns
    -------
    LayoutProblem
        CPU layout problem with edge tensor shape ``[2, E]``.
    """
    edges = torch.tensor(
        [[index for index in range(num_nodes - 1)], [index + 1 for index in range(num_nodes - 1)]],
        dtype=torch.long,
    )
    return LayoutProblem(
        edge_index=edges,
        num_nodes=num_nodes,
        node_sizes=torch.full((num_nodes, 2), 10.0),
        direction="TB",
        seed=42,
    )


def _score_by_first_x(
    problem: LayoutProblem,
    pos: torch.Tensor,
) -> Tuple[Tuple[int, float], float]:
    """Return a deterministic fake V3 score from the first x-coordinate.

    Parameters
    ----------
    problem : LayoutProblem
        Scored problem, unused beyond signature compatibility.
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.

    Returns
    -------
    tuple[tuple[int, float], float]
        Neutral severe-G6 key and fake higher-is-better score.
    """
    del problem
    return (1, -0.0), float(pos[0, 0].item())


def test_anytime_native_coarsest_skips_native_when_budget_has_no_return_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper returns the priced fallback when no native arm can be admitted."""
    fallback = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    called = {"native": False}

    def fake_fallback(problem: LayoutProblem, seed: int) -> torch.Tensor:
        """Return a deterministic fallback tensor."""
        del problem, seed
        return fallback

    def forbidden_native(*args: object, **kwargs: object) -> torch.Tensor:
        """Fail if the budget gate launches native without reserve."""
        del args, kwargs
        called["native"] = True
        raise AssertionError("native should not run without return reserve")

    monkeypatch.setattr(coarsest, "_stress_sgd_fallback", fake_fallback)
    monkeypatch.setattr(coarsest, "_score_v3_position", _score_by_first_x)
    monkeypatch.setattr(coarsest, "layout_dagua_native_pipeline", forbidden_native)

    started = time.perf_counter()
    actual = coarsest.anytime_native_coarsest(
        _chain_problem(),
        LayoutConfig(),
        time_budget_s=0.001,
        seed=42,
    )
    elapsed = time.perf_counter() - started

    assert torch.equal(actual, fallback)
    assert called == {"native": False}
    assert elapsed < 0.5


def test_anytime_native_coarsest_falls_back_when_native_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native failure after fallback scoring returns the priced fallback."""
    fallback = torch.zeros((4, 2), dtype=torch.float32)

    def fake_fallback(problem: LayoutProblem, seed: int) -> torch.Tensor:
        """Return a deterministic fallback tensor."""
        del problem, seed
        return fallback

    def raising_native(*args: object, **kwargs: object) -> torch.Tensor:
        """Simulate a portfolio failure after admission."""
        del args, kwargs
        raise RuntimeError("native failed")

    monkeypatch.setattr(coarsest, "_stress_sgd_fallback", fake_fallback)
    monkeypatch.setattr(coarsest, "_score_v3_position", _score_by_first_x)
    monkeypatch.setattr(coarsest, "layout_dagua_native_pipeline", raising_native)

    actual = coarsest.anytime_native_coarsest(
        _chain_problem(),
        LayoutConfig(),
        time_budget_s=2.0,
        seed=42,
    )

    assert torch.equal(actual, fallback)


def test_anytime_native_coarsest_is_monotone_against_stress_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lower-scored native completion cannot replace the stress fallback."""
    fallback = torch.tensor(
        [[5.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        dtype=torch.float32,
    )
    worse_native = torch.zeros((4, 2), dtype=torch.float32)

    def fake_fallback(problem: LayoutProblem, seed: int) -> torch.Tensor:
        """Return a higher-scored fallback tensor."""
        del problem, seed
        return fallback

    def fake_native(*args: object, **kwargs: object) -> torch.Tensor:
        """Return a lower-scored native tensor."""
        del args, kwargs
        return worse_native

    monkeypatch.setattr(coarsest, "_stress_sgd_fallback", fake_fallback)
    monkeypatch.setattr(coarsest, "_score_v3_position", _score_by_first_x)
    monkeypatch.setattr(coarsest, "layout_dagua_native_pipeline", fake_native)

    actual = coarsest.anytime_native_coarsest(
        _chain_problem(),
        LayoutConfig(),
        time_budget_s=2.0,
        seed=42,
    )

    assert torch.equal(actual, fallback)


def test_anytime_native_coarsest_is_byte_deterministic_with_seed_42() -> None:
    """Two fallback-priced runs with seed 42 produce byte-identical tensors."""
    problem = _chain_problem(6)
    config = LayoutConfig()

    first = coarsest.anytime_native_coarsest(problem, config, time_budget_s=0.001, seed=42)
    second = coarsest.anytime_native_coarsest(problem, config, time_budget_s=0.001, seed=42)

    assert torch.equal(first, second)
