"""Shared deterministic fixtures for RULER V4 property families."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterator

import pytest
import torch

from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile
from dagua.eval.ruler_v4.contracts import CONTRACTS
from dagua.eval.ruler_v4.headline import HeadlineProfile
from dagua.eval.ruler_v4.ingestion import ingest
from dagua.eval.ruler_v4.scene import (
    DrawingScene,
    GraphSemantics,
    ObservationProfile,
    Route,
    Scene,
    StyleContract,
    ValidScene,
)
from dagua.eval.ruler_v4.score import ScoringProfiles
from dagua.eval.ruler_v4.weights import (
    GATE_DIAGNOSTIC_FACETS,
    REQUIRED_PRIOR_FLOOR_FACETS,
    SubtermWeight,
    WeightTable,
)

_PROPERTY_SEED = 1729


@pytest.fixture(autouse=True)
def fixed_property_seed() -> Iterator[None]:
    """Run every property under a fixed torch seed and restore caller state.

    Yields
    ------
    None
        Control while the deterministic seed is active.
    """

    state = torch.random.get_rng_state()
    torch.manual_seed(_PROPERTY_SEED)
    yield
    torch.random.set_rng_state(state)


def _group_for_facet(facet_id: str) -> str:
    """Map one facet into the frozen reporting rollup used by test weights.

    Parameters
    ----------
    facet_id : str
        Frozen facet id.

    Returns
    -------
    str
        Reporting-group id.
    """

    number = int(facet_id[1:].rstrip("ab"))
    if number <= 6 or number in {9, 14}:
        return "S"
    if number in {7, 8, 10, 11, 12, 13, 15, 16, 41}:
        return "X"
    if 17 <= number <= 24 or number == 42:
        return "L"
    if 25 <= number <= 30:
        return "C"
    if 31 <= number <= 34:
        return "F"
    if 35 <= number <= 37:
        return "W"
    if number == 38:
        return "M"
    if number == 39:
        return "P"
    return "T"


def build_full_weight_table() -> WeightTable:
    """Build a complete explicit fixed-prior table for scorer properties.

    Returns
    -------
    WeightTable
        All 91 scored rows, with gates-report diagnostics carried at zero.
    """

    entries = tuple(
        SubtermWeight(
            subterm_id=subterm_id,
            facet_id=facet_id,
            group=_group_for_facet(facet_id),
            weight=0.0 if facet_id in GATE_DIAGNOSTIC_FACETS else 1.0,
            prior_driven=facet_id in REQUIRED_PRIOR_FLOOR_FACETS,
            diagnostic=facet_id in GATE_DIAGNOSTIC_FACETS,
        )
        for facet_id, contract in CONTRACTS.items()
        for subterm_id in contract.scored_subterms
    )
    return WeightTable(
        entries=entries,
        d_power=20,
        prior_floors={facet_id: 1.0 for facet_id in REQUIRED_PRIOR_FLOOR_FACETS},
    )


@pytest.fixture
def full_weight_table() -> WeightTable:
    """Return complete deterministic weights for top-level score properties.

    Returns
    -------
    WeightTable
        Frozen complete test table.
    """

    return build_full_weight_table()


@pytest.fixture
def scoring_profiles() -> ScoringProfiles:
    """Return explicit deterministic p-mean and headline profiles.

    Returns
    -------
    ScoringProfiles
        Versioned property-test profiles.
    """

    return ScoringProfiles(
        composition=CompositionProfile(CompositionFamily.P_MEAN, power=2.0),
        headline=HeadlineProfile(index_span=100.0, loss_scale=1.0, version="property-1"),
        measurement_version="v4-measurement-property",
        policy_version="v4-policy-property",
        alpha_grid_index=1,
    )


@pytest.fixture
def scorable_scene(semantic_scene: Scene) -> Scene:
    """Add a complete declared-tree block to the broad semantic scene.

    Parameters
    ----------
    semantic_scene : Scene
        Shared eight-node semantic fixture.

    Returns
    -------
    Scene
        Immutable scene for which applicable facets are VALUE and the rest NA.
    """

    graph = replace(
        semantic_scene.graph,
        tree_parents=(None, 0, 0, 1, 2, 3, 4, 5),
        tree_depths=(0, 1, 1, 2, 2, 3, 3, 4),
        tree_layout="layered",
        flow_axis=(1.0, 0.0),
        ordered_children={0: (1, 2)},
    )
    return replace(semantic_scene, graph=graph)


def reingest_scaled(scene: Scene, factor: float, *, position_only: bool) -> Scene:
    """Reingest a deterministic scale transform through the public front door.

    Parameters
    ----------
    scene : Scene
        Validated source scene.
    factor : float
        Positive binary-exact scale factor.
    position_only : bool
        Keep primitives fixed when true; otherwise re-express scene units.

    Returns
    -------
    Scene
        Validated transformed scene.
    """

    routes = tuple(
        Route(route.edge_index, route.points * factor, route.kind) for route in scene.routes
    )
    style = scene.style
    if not position_only:
        style = replace(
            style,
            coordinate_scale=style.coordinate_scale * factor,
            edge_stroke_widths=tuple(value * factor for value in style.edge_stroke_widths),
        )
    result = ingest(
        scene.graph,
        DrawingScene(scene.positions * factor, routes, scene.z_order),
        style,
        scene.profile,
    )
    assert isinstance(result, ValidScene)
    return result.scene


def build_crossing_scene(top_x: float, top_y: float) -> Scene:
    """Build two nonincident routes near the crossing event manifold.

    Parameters
    ----------
    top_x, top_y : float
        Coordinates of the upper endpoint of the second edge.

    Returns
    -------
    Scene
        Validated deterministic four-node scene.
    """

    positions = torch.tensor(
        [[-2.0, 0.0], [2.0, 0.0], [0.0, -2.0], [top_x, top_y]],
        dtype=torch.float64,
    )
    edges = ((0, 1), (2, 3))
    routes = tuple(
        Route(index, torch.stack((positions[source], positions[target])))
        for index, (source, target) in enumerate(edges)
    )
    result = ingest(
        GraphSemantics(
            node_ids=("a", "b", "c", "d"),
            edges=edges,
            required_primitives=frozenset({"nodes", "routes"}),
        ),
        DrawingScene(positions, routes),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes"})),
    )
    assert isinstance(result, ValidScene)
    return result.scene
