"""Property family: K12 crossing addition is strictly monotone."""

from __future__ import annotations

from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile, compose
from dagua.eval.ruler_v4.edges import U07
from dagua.eval.ruler_v4.weight_table import SubtermWeight, WeightTable

from .conftest import build_crossing_scene


def test_adding_one_crossing_strictly_worsens_u07_and_composite() -> None:
    """An endpoint sweep that adds one crossing raises raw count and loss."""

    crossing_free = U07(build_crossing_scene(top_x=0.0, top_y=-0.001))
    crossed = U07(build_crossing_scene(top_x=0.0, top_y=0.001))
    weights = WeightTable(
        entries=(
            SubtermWeight("U7.base", "U07", "X", 1.0),
            SubtermWeight("U7.tail", "U07", "X", 1.0),
        ),
        d_power=0,
    )
    profile = CompositionProfile(CompositionFamily.P_MEAN, power=2.0)
    baseline = compose({"U07": crossing_free}, weights, profile)
    degraded = compose({"U07": crossed}, weights, profile)

    assert crossed.raw["crossing_count"] == crossing_free.raw["crossing_count"] + 1
    assert crossing_free.value is not None
    assert crossed.value is not None
    assert crossed.value > crossing_free.value
    assert degraded.l_total > baseline.l_total


def _certified_planar_scene(bowtie: bool):
    """Ingest a certified-planar 4-cycle, optionally drawn with a crossing.

    Parameters
    ----------
    bowtie : bool
        Swap two adjacent corners so exactly one pair of cycle edges
        crosses while the input planarity certificate is unchanged.

    Returns
    -------
    Scene
        Validated static scene with U41 applicable.
    """

    import torch

    from dagua.eval.ruler_v4.ingestion import ingest
    from dagua.eval.ruler_v4.scene import (
        DrawingScene,
        GraphSemantics,
        ObservationProfile,
        Route,
        StyleContract,
        ValidScene,
    )

    corners = [[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0]]
    if bowtie:
        corners[2], corners[3] = corners[3], corners[2]
    positions = torch.tensor(corners, dtype=torch.float64)
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    routes = tuple(
        Route(index, torch.stack((positions[source], positions[target])))
        for index, (source, target) in enumerate(edges)
    )
    result = ingest(
        GraphSemantics(
            node_ids=("a", "b", "c", "d"),
            edges=edges,
            planarity_certificate={"planar": True},
        ),
        DrawingScene(positions, routes),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes"})),
    )
    assert isinstance(result, ValidScene)
    return result.scene


def test_combined_crossing_and_face_metamorph_strictly_worsens_composite() -> None:
    """Combined applicability: a crossing on a certified-planar row worsens
    the crossing-plus-face composite when BOTH facets degrade together.

    On this bowtie fixture U41 worsens alongside U07, so a monotonicity
    inversion in either facet fails the assertion -- but the 5.5 r3
    exchange-rate disease (face relief paying for a priced crossing) needs a
    fixture where the added crossing genuinely REDUCES face debt, which this
    is not (DISCREPANCIES entry 37)."""

    from dagua.eval.ruler_v4.packing import U41
    from dagua.eval.ruler_v4.scene import ResultState

    flat_u07 = U07(_certified_planar_scene(bowtie=False))
    flat_u41 = U41(_certified_planar_scene(bowtie=False))
    crossed_u07 = U07(_certified_planar_scene(bowtie=True))
    crossed_u41 = U41(_certified_planar_scene(bowtie=True))
    for result in (flat_u07, flat_u41, crossed_u07, crossed_u41):
        assert result.state is ResultState.VALUE

    assert crossed_u07.raw["crossing_count"] == flat_u07.raw["crossing_count"] + 1

    weights = WeightTable(
        entries=(
            SubtermWeight("U7.base", "U07", "X", 1.0),
            SubtermWeight("U7.tail", "U07", "X", 1.0),
            *(SubtermWeight(subterm_id, "U41", "X", 1.0) for subterm_id in flat_u41.subterms),
        ),
        d_power=0,
    )
    profile = CompositionProfile(CompositionFamily.P_MEAN, power=2.0)
    baseline = compose({"U07": flat_u07, "U41": flat_u41}, weights, profile)
    degraded = compose({"U07": crossed_u07, "U41": crossed_u41}, weights, profile)
    assert degraded.l_total > baseline.l_total
