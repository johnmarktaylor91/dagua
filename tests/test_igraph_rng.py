"""Regression tests for igraph-compatible RNG bridges."""

from __future__ import annotations

import pytest

from dagua.layout.ops._igraph_rng import IgraphMT19937, IgraphPCG32


def test_igraph_pcg32_matches_golden_raw_words() -> None:
    """PCG32 raw words should match the igraph default generator path."""
    rng = IgraphPCG32(seed=42)

    draws = [rng.get_u32() for _ in range(10)]

    assert draws == [
        589627368,
        2336806640,
        3609466837,
        4225723592,
        4227023364,
        3175032210,
        3962267976,
        104173945,
        517798263,
        2486465614,
    ]


def test_igraph_pcg32_matches_golden_high_level_draws() -> None:
    """PCG32 high-level extraction should match igraph random.c wrappers."""
    float_rng = IgraphPCG32(seed=42)
    int_rng = IgraphPCG32(seed=42)

    floats = [float_rng.random() for _ in range(5)]
    ints = [int_rng.randint(0, 100) for _ in range(10)]

    assert floats == pytest.approx(
        [
            0.1372833197340555,
            0.8403944871350837,
            0.984180570752184,
            0.9225374031868425,
            0.12055930299193718,
        ],
        rel=0.0,
        abs=0.0,
    )
    assert ints == [13, 54, 84, 99, 99, 74, 93, 2, 12, 58]


def test_igraph_pcg32_advance_matches_repeated_draws() -> None:
    """PCG32 jump-ahead should land on the same stream position as iteration."""
    iterated = IgraphPCG32(seed=137)
    jumped = IgraphPCG32(seed=137)

    expected = [iterated.get_u32() for _ in range(25)][-1]
    jumped.advance(24)

    assert jumped.get_u32() == expected


def test_igraph_mt19937_matches_golden_raw_words() -> None:
    """MT19937 raw words should preserve igraph's legacy generator stream."""
    rng = IgraphMT19937(seed=42)

    draws = [rng.get_u32() for _ in range(10)]

    assert draws == [
        1608637542,
        3421126067,
        4083286876,
        787846414,
        3143890026,
        3348747335,
        2571218620,
        2563451924,
        670094950,
        1914837113,
    ]


def test_igraph_mt19937_matches_golden_high_level_draws() -> None:
    """MT19937 high-level extraction should match igraph random.c wrappers."""
    float_rng = IgraphMT19937(seed=42)
    int_rng = IgraphMT19937(seed=42)

    floats = [float_rng.random() for _ in range(5)]
    ints = [int_rng.randint(0, 100) for _ in range(10)]

    assert floats == pytest.approx(
        [
            0.37454011449509816,
            0.9507143116051877,
            0.7319939385120968,
            0.5986584864083793,
            0.156018638621511,
        ],
        rel=0.0,
        abs=0.0,
    )
    assert ints == [37, 80, 96, 18, 73, 78, 60, 60, 15, 45]


def test_igraph_mt19937_zero_seed_uses_legacy_default_seed() -> None:
    """MT19937 seed zero should map to igraph's legacy seed 4357."""
    zero_seed = IgraphMT19937(seed=0)
    legacy_seed = IgraphMT19937(seed=4357)

    assert [zero_seed.get_u32() for _ in range(10)] == [legacy_seed.get_u32() for _ in range(10)]
