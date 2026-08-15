"""Property family: catastrophic bottlenecks resist mean buyback."""

from __future__ import annotations

import pytest

from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile, compose
from dagua.eval.ruler_v4.scene import FacetResult, value_result
from dagua.eval.ruler_v4.weight_table import SubtermWeight, WeightTable


def _weights() -> WeightTable:
    """Return one equal explicit mass in each synthetic group.

    Returns
    -------
    WeightTable
        Fixed three-group table.
    """

    return WeightTable(
        entries=(
            SubtermWeight("S.loss", "S1", "S", 1.0),
            SubtermWeight("X.loss", "X1", "X", 1.0),
            SubtermWeight("L.loss", "L1", "L", 1.0),
        ),
        d_power=0,
    )


def _losses(structure: float, edges: float, legibility: float) -> dict[str, FacetResult]:
    """Build three bounded independent group values.

    Parameters
    ----------
    structure, edges, legibility : float
        Synthetic group defects in ``[0, 1]``.

    Returns
    -------
    dict[str, FacetResult]
        Mapping accepted by composition.
    """

    return {
        "S1": value_result(structure, {"S.loss": structure}),
        "X1": value_result(edges, {"X.loss": edges}),
        "L1": value_result(legibility, {"L.loss": legibility}),
    }


def test_catastrophic_bottleneck_cannot_be_bought_back_by_better_mean() -> None:
    """A slightly better mean cannot hide one catastrophic group."""

    profile = CompositionProfile(
        CompositionFamily.MEAN_SOFT_BOTTLENECK,
        bottleneck_mix=0.4,
        bottleneck_temperature=0.05,
        group_allowances={"S": 0.1, "X": 0.1, "L": 0.1},
    )
    sacrificed = compose(_losses(1.0, 0.0, 0.0), _weights(), profile)
    diffuse = compose(_losses(0.34, 0.34, 0.34), _weights(), profile)

    assert sacrificed.l_mean < diffuse.l_mean
    assert sacrificed.l_total > diffuse.l_total
    assert sacrificed.l_bottleneck is not None
    assert diffuse.l_bottleneck is not None
    assert sacrificed.l_bottleneck > diffuse.l_bottleneck


def test_shipped_p_mean_default_is_noncompensatory_at_fixed_mean() -> None:
    """P2 gap: the R3-DR default family had no non-compensation property.

    At any p > 1 and fixed l_mean, concentrating defect into one row
    strictly raises l_total; a pure mean (the p = 1 boundary) cannot
    distinguish the two, which is exactly what makes this falsifiable.
    """

    concentrated = _losses(1.0, 0.01, 0.01)
    diffuse = _losses(0.34, 0.34, 0.34)
    for power in (1.5, 2.0, 4.0):
        profile = CompositionProfile(CompositionFamily.P_MEAN, power=power)
        sacrificed = compose(concentrated, _weights(), profile)
        spread = compose(diffuse, _weights(), profile)
        assert sacrificed.l_mean == pytest.approx(spread.l_mean)
        assert sacrificed.l_total > spread.l_total

    linear = CompositionProfile(CompositionFamily.P_MEAN, power=1.0)
    assert compose(concentrated, _weights(), linear).l_total == pytest.approx(
        compose(diffuse, _weights(), linear).l_total
    )


def test_every_coordinate_bump_strictly_worsens_both_families() -> None:
    """P2 gap: the single-coordinate non-strict check let a composition
    ignore a row entirely. Every positive-mass coordinate must bite."""

    base = (0.2, 0.3, 0.4)
    profiles = (
        CompositionProfile(CompositionFamily.P_MEAN, power=2.0),
        CompositionProfile(
            CompositionFamily.MEAN_SOFT_BOTTLENECK,
            bottleneck_mix=0.4,
            bottleneck_temperature=0.05,
            group_allowances={"S": 0.1, "X": 0.1, "L": 0.1},
        ),
    )
    for profile in profiles:
        baseline = compose(_losses(*base), _weights(), profile)
        for coordinate in range(3):
            bumped = list(base)
            bumped[coordinate] += 0.05
            degraded = compose(_losses(*bumped), _weights(), profile)
            assert degraded.l_total > baseline.l_total
