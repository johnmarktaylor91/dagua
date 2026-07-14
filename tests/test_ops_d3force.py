"""Unit pins for d3-force-compatible ops."""

from __future__ import annotations

import torch

from dagua.layout.ops.d3force import d3force_lcg_values, d3force_phyllotaxis_positions


def test_d3force_lcg_matches_reference_first_20_values() -> None:
    """Pin d3-force's LCG sequence.

    Returns
    -------
    None
        The first 20 values must match a Node reference script bit-for-bit.
    """
    expected = [
        0.23645552527159452,
        0.3692706737201661,
        0.5042420323006809,
        0.7048832636792213,
        0.05054362863302231,
        0.3695183543022722,
        0.7747629624791443,
        0.556188570568338,
        0.0164932357147336,
        0.6392460397910327,
        0.2504511415027082,
        0.4223777682054788,
        0.5906901974231005,
        0.8369336591567844,
        0.23507591942325234,
        0.980845961952582,
        0.8608870944008231,
        0.32687550294212997,
        0.6826027217321098,
        0.5314591128844768,
    ]
    assert d3force_lcg_values(seed=1, count=20) == expected


def test_d3force_phyllotaxis_matches_reference_initial_nodes() -> None:
    """Pin d3-force's missing-position initialization spiral.

    Returns
    -------
    None
        First six phyllotaxis positions must match d3-force.
    """
    expected = torch.tensor(
        [
            [7.0710678118654755, 0.0],
            [-9.03088751750192, 8.273032735715967],
            [1.3823220809823638, -15.750847141167634],
            [11.382848792909423, 14.846910566099618],
            [-20.88892748977138, -3.694957148205299],
            [19.78781566111266, -12.587388583889217],
        ],
        dtype=torch.float64,
    )
    actual = d3force_phyllotaxis_positions(6)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
