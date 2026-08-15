"""Property family: catastrophic bottlenecks resist mean buyback."""

from __future__ import annotations

from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile, compose
from dagua.eval.ruler_v4.scene import FacetResult, value_result
from dagua.eval.ruler_v4.weights import SubtermWeight, WeightTable


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
