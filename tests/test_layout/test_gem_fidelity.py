"""Regression tests for GEM OGDF fidelity helpers."""

from __future__ import annotations

import torch

from dagua.layout.ops.gem import (
    _glibc_rand_values,
    _ogdf_gem_rng_seed,
    _ogdf_permutation,
    _ogdf_runner_initial_positions,
    _ogdf_uniform_int,
    _OgdfMinStdRand,
)


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


def test_ogdf_minstd_permutation_matches_cpp_fixture() -> None:
    """Verify OGDF GEM node permutation uses the C++ minstd schedule.

    Returns
    -------
    None
        The assertion validates seed translation and ``SList::permute`` order.
    """
    rng = _OgdfMinStdRand(_ogdf_gem_rng_seed(42))

    assert _ogdf_permutation(5, rng) == [3, 4, 1, 0, 2]


def test_ogdf_zero_disturbance_consumes_rng() -> None:
    """Verify ``uniform_int_distribution(0, 0)`` advances minstd state.

    Returns
    -------
    None
        The assertion guards GEM's post-init RNG stream against drift.
    """
    rng = _OgdfMinStdRand(123)

    assert _ogdf_uniform_int(rng, 0, 0) == 0
    assert rng.next() == 985_676_192
