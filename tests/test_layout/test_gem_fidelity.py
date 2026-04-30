"""Regression tests for GEM OGDF fidelity helpers."""

from __future__ import annotations

import torch

from dagua.layout.ops.gem import _glibc_rand_values, _ogdf_runner_initial_positions


def test_glibc_rand_values_match_seed_42_modulo_sequence() -> None:
    """Verify the glibc ``rand()`` reproducer against known seed-42 output.

    Returns
    -------
    None
        The assertion validates the deterministic sequence.
    """
    values = _glibc_rand_values(seed=42, count=6)

    assert [value % 1000 for value in values] == [166, 740, 881, 241, 12, 758]


def test_ogdf_runner_initial_positions_match_seed_42_fixture() -> None:
    """Verify OGDF runner initial positions use interleaved glibc values.

    Returns
    -------
    None
        The assertion validates the expected ``[N, 2]`` initialization tensor.
    """
    positions = _ogdf_runner_initial_positions(
        num_nodes=3,
        seed=42,
        device=torch.device("cpu"),
    )

    expected = torch.tensor(
        [[16.6, 74.0], [88.1, 24.1], [1.2, 75.8]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(positions, expected)
