"""Tests for Lane C visual parity sweep dry-runs."""

from __future__ import annotations

import pytest

from scripts.visual_parity import sweep


def test_one_dimensional_sweep_prints_argmin_and_no_worktree() -> None:
    """Dry-run sweep output should expose argmin without worktree candidates.

    Returns
    -------
    None
        The test asserts table content.
    """

    results = sweep.run_sweep("edge.arrow.length", [0.8, 1.0, 1.2], ["case_a", "case_b"])
    table = sweep.render_table(results)

    assert "argmin" in table
    assert "execution: in-process render overrides; no worktree candidates" in table
    assert table.count("yes") >= 1


def test_grid_sweep_is_limited_to_7_by_7() -> None:
    """Multi-knob sweeps should require a bounded declared grid.

    Returns
    -------
    None
        The test asserts F16 grid protection.
    """

    with pytest.raises(ValueError, match="7x7"):
        sweep.run_grid_sweep(
            "edge.arrow.length",
            [float(index) for index in range(8)],
            "edge.arrow.width",
            [1.0],
            ["case_a"],
        )
