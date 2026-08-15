"""Worked examples for declared-cluster facet contracts."""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import pytest
import torch

from dagua.eval.ruler_v4.clusters import U25, U26, U27, U28, U29, U30
from dagua.eval.ruler_v4.ingestion import ingest
from dagua.eval.ruler_v4.scene import (
    DrawingScene,
    GraphSemantics,
    ObservationProfile,
    ResultState,
    Route,
    Scene,
    StyleContract,
    ValidScene,
)


def _scene(
    positions: torch.Tensor,
    clusters: Mapping[str, Tuple[int, ...]],
    parents: Optional[Mapping[str, str]] = None,
    cluster_labels_visible: bool = False,
) -> Scene:
    """Ingest one clustered worked-example scene.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    clusters : mapping[str, tuple[int, ...]]
        Declared cluster memberships.
    parents : mapping[str, str]
        Optional child-to-parent hierarchy.
    cluster_labels_visible : bool
        Whether the profile declares cluster-label rendering.

    Returns
    -------
    Scene
        Validated clustered scene.
    """

    edges = tuple((index, index + 1) for index in range(positions.shape[0] - 1))
    graph = GraphSemantics(
        tuple(f"n{index}" for index in range(positions.shape[0])),
        edges,
        clusters=clusters,
        cluster_parents=dict(parents or {}),
    )
    routes = tuple(
        Route(index, torch.stack((positions[source], positions[target])))
        for index, (source, target) in enumerate(edges)
    )
    channels = {"nodes", "routes"}
    if cluster_labels_visible:
        channels.add("cluster_labels")
    result = ingest(
        graph,
        DrawingScene(positions, routes),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset(channels)),
    )
    assert isinstance(result, ValidScene)
    return result.scene


def test_u25_uniform_scale_preserves_shape_defect() -> None:
    """U25's scale-neutral contract gives a homothetic pair the same defect."""

    compact = _scene(torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]), {"c": (0, 1, 2)})
    spread = _scene(torch.tensor([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]]), {"c": (0, 1, 2)})
    compact_result = U25(compact)
    spread_result = U25(spread)
    assert compact_result.value is not None
    assert spread_result.value is not None
    assert compact_result.value == pytest.approx(0.0, abs=0.0)
    assert spread_result.value == pytest.approx(0.0, abs=0.0)


def test_u26_separated_clusters_score_better_than_interleaved_twin() -> None:
    """U26's community fixture prefers separated declared clusters."""

    clusters = {"a": (0, 1, 2), "b": (3, 4, 5)}
    separated = _scene(
        torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [8.0, 0.0], [8.0, 1.0], [9.0, 0.0]]),
        clusters,
    )
    interleaved = _scene(
        torch.tensor([[0.0, 0.0], [8.0, 1.0], [1.0, 0.0], [8.0, 0.0], [0.0, 1.0], [9.0, 0.0]]),
        clusters,
    )
    separated_result = U26(separated)
    interleaved_result = U26(interleaved)
    assert separated_result.value is not None
    assert interleaved_result.value is not None
    assert separated_result.value == pytest.approx(0.0, abs=0.0)
    # U26 contract golden 4: interleaved communities "score D^i near 1" and
    # "improve monotonically as the declared communities are separated."
    assert interleaved_result.subterms["U26.i"] > separated_result.subterms["U26.i"]
    assert separated_result.value < interleaved_result.value


def test_u27_nonmember_nodes_and_routes_outside_region_have_no_intrusion() -> None:
    """U27's clean containment fixture has zero foreign-node and route intrusion."""

    scene = _scene(
        torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [8.0, 0.0], [8.0, 1.0], [9.0, 0.0]]),
        {"a": (0, 1, 2), "b": (3, 4, 5)},
    )
    result = U27(scene, None)
    assert result.state is ResultState.NA
    assert result.reason == "alpha_grid_unselected"
    lower, upper = result.raw["grid_envelope"]
    assert lower <= upper
    # U27 contract golden 4: "(ii) = 0 for every member interior to its cluster";
    # golden 5 likewise scores only a foreign route routed through the cluster.
    assert upper == pytest.approx(0.0, abs=0.0)


def test_u28_nested_parent_contains_child_without_overflow() -> None:
    """U28's nested hierarchy fixture has zero parent-child containment debt."""

    positions = torch.tensor(
        [
            [-4.0, -4.0],
            [4.0, -4.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [-4.0, 4.0],
            [4.0, 4.0],
        ]
    )
    scene = _scene(
        positions,
        {"child": (2, 3, 4), "parent": (0, 1, 2, 3, 4, 5, 6)},
        {"child": "parent"},
    )
    result = U28(scene, None)
    assert result.state is ResultState.NA
    assert result.reason == "alpha_grid_unselected"
    lower, upper = result.raw["grid_envelope"]
    assert lower <= upper
    assert (lower, upper) == pytest.approx((1.0, 1.0), abs=0.0)


def test_u29_square_cluster_has_exact_zero_shape_debt() -> None:
    """U29's square cluster has unit aspect ratio and exact zero defect."""

    positions = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]])
    result = U29(_scene(positions, {"square": (0, 1, 2, 3, 4)}))
    assert result.state is ResultState.VALUE
    assert result.value == pytest.approx(0.0, abs=0.0)


def test_u30_absent_cluster_label_channel_is_typed_na() -> None:
    """U30's applicability golden is NA when cluster-label rendering is undeclared."""

    positions = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    result = U30(_scene(positions, {"square": (0, 1, 2, 3)}), None)
    assert result.state is ResultState.NA
    assert result.reason == "no_declared_cluster_labels"


def test_u30_derived_label_is_exactly_at_declared_padding() -> None:
    """U30's derived label scores exact zero on the declared padding row."""

    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    scene = _scene(
        positions,
        {"c": (0, 1, 2)},
        cluster_labels_visible=True,
    )
    result = U30(scene, 1)
    assert result.state is ResultState.VALUE
    # U30 contract golden 3: a label exactly at declared padding scores zero.
    assert result.subterms["U30.ii"] == pytest.approx(0.0, abs=0.0)
    label = scene.cluster_label_boxes["c"]
    assert float(label.center[0]) == pytest.approx(1.0, abs=0.0)


def _traced_cluster_scene() -> Scene:
    """Build the clusters traced-seam fixture: two communities plus strays.

    Seeded noise off a two-community layout with a shared parent cluster,
    two unclustered stray nodes, and declared cluster labels, so the
    separation (U26), containment (U27), hierarchy (U28), and label (U30)
    channels all carry nonzero defects.

    Returns
    -------
    Scene
        Validated clustered scene.
    """

    generator = torch.Generator().manual_seed(0)
    base = torch.tensor(
        [
            [0.0, 0.0],
            [0.9, 0.1],
            [0.2, 1.1],
            [1.3, 0.9],
            [4.2, 0.3],
            [5.1, -0.2],
            [4.6, 1.2],
            [5.4, 0.8],
            [2.6, 2.4],
            [2.4, -1.6],
        ],
        dtype=torch.float64,
    )
    positions = base + 0.13 * torch.randn(base.shape, generator=generator, dtype=torch.float64)
    edges = tuple((index, index + 1) for index in range(9)) + ((0, 3), (4, 7))
    graph = GraphSemantics(
        tuple(f"n{index}" for index in range(10)),
        edges,
        clusters={"a": (0, 1, 2, 3), "b": (4, 5, 6, 7), "top": tuple(range(8))},
        cluster_parents={"a": "top", "b": "top"},
    )
    routes = tuple(
        Route(index, torch.stack((positions[source], positions[target])))
        for index, (source, target) in enumerate(edges)
    )
    result = ingest(
        graph,
        DrawingScene(positions, routes),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes", "cluster_labels"})),
    )
    assert isinstance(result, ValidScene)
    return result.scene


def _score_traced(scene: Scene):
    """Score one scene through the traced surrogate seam."""

    from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile
    from dagua.eval.ruler_v4.contracts import CONTRACTS
    from dagua.eval.ruler_v4.headline import HeadlineProfile
    from dagua.eval.ruler_v4.score import ScoringProfiles
    from dagua.eval.ruler_v4.surrogate.traced import score_scene_soft
    from dagua.eval.ruler_v4.weight_table import (
        GATE_DIAGNOSTIC_FACETS,
        REQUIRED_PRIOR_FLOOR_FACETS,
        ParameterProvenance,
        SubtermWeight,
        WeightTable,
    )

    entries = tuple(
        SubtermWeight(
            subterm_id=subterm_id,
            facet_id=facet_id,
            group=facet_id[:3],
            weight=0.0 if facet_id in GATE_DIAGNOSTIC_FACETS else 1.0,
            prior_driven=facet_id in REQUIRED_PRIOR_FLOOR_FACETS,
            diagnostic=facet_id in GATE_DIAGNOSTIC_FACETS,
            provenance_class=(
                None
                if facet_id in GATE_DIAGNOSTIC_FACETS
                else "preregistered_prior"
                if facet_id in REQUIRED_PRIOR_FLOOR_FACETS
                else "contract_frozen"
            ),
        )
        for facet_id, contract in CONTRACTS.items()
        for subterm_id in contract.scored_subterms
    )
    table = WeightTable(
        entries=entries,
        d_power=20,
        prior_floors={facet_id: 1.0 for facet_id in REQUIRED_PRIOR_FLOOR_FACETS},
    )
    profiles = ScoringProfiles(
        composition=CompositionProfile(CompositionFamily.P_MEAN, power=2.0),
        headline=HeadlineProfile(index_span=100.0, loss_scale=1.0, version="probe"),
        measurement_version="probe-measurement",
        policy_version="probe-policy",
        alpha_grid_index=1,
        parameter_provenance={
            "composition.power": ParameterProvenance("preregistered_prior"),
            "headline.index_span": ParameterProvenance("contract_frozen"),
            "headline.loss_scale": ParameterProvenance("preregistered_prior"),
            "alpha_grid_index": ParameterProvenance("preregistered_prior"),
        },
    )
    return score_scene_soft(scene, table, profiles)


@pytest.fixture(scope="module")
def cluster_traced():
    """Score the clusters fixture once through the traced seam (expensive)."""

    scene = _traced_cluster_scene()
    return scene, _score_traced(scene)


def test_cluster_traced_rows_carry_live_position_gradients(cluster_traced) -> None:
    """Traced cluster geometry rows differentiate against positions.

    Live rows on this fixture: separation margins (U26.i), community
    stratum contrasts (U26.iii), foreign-node intrusion (U27.i), parent
    coverage economy (U28.ii), sibling-overlap (U28.iii), and the derived
    cluster-label containment (U30.i, live through the region geometry
    even while the label box itself is an input-owned constant).
    """

    import torch as _torch

    _, traced = cluster_traced
    live_rows = ("U26.i", "U26.iii", "U27.i", "U28.ii", "U28.iii", "U30.i")
    for row in live_rows:
        tensor = traced.traced_subterms[row]
        assert tensor.requires_grad, row
        (gradient,) = _torch.autograd.grad(
            tensor, traced.positions, retain_graph=True, allow_unused=True
        )
        assert gradient is not None, row
        assert float(_torch.linalg.vector_norm(gradient)) > 0.0, row
    # Honest flats on this fixture, traced but exactly stationary:
    # U25.headline and U29.headline sit at their exact-zero defects
    # (anchored zeros), U27.ii at zero member escape, U28.i at zero
    # overflow, U30.ii at zero padding debt; U27.iii's route-intrusion
    # fade is saturated (every foreign route fully inside or outside the
    # 0.25 band, where the smooth fade's derivative is exactly zero).
    for row in ("U25.headline", "U29.headline", "U27.ii", "U27.iii", "U28.i", "U30.ii"):
        tensor = traced.traced_subterms[row]
        assert tensor.requires_grad, row
        (gradient,) = _torch.autograd.grad(
            tensor, traced.positions, retain_graph=True, allow_unused=True
        )
        assert gradient is not None, row
        assert float(_torch.linalg.vector_norm(gradient)) == 0.0, row
    assert traced.soft.l_total.requires_grad


def test_cluster_traced_values_match_untraced_evaluation(cluster_traced) -> None:
    """Traced cluster subterm values agree with the exact un-traced facets."""

    scene, traced = cluster_traced
    exact = {}
    for facet, needs_grid in (
        (U25, False),
        (U26, False),
        (U27, True),
        (U28, True),
        (U29, False),
        (U30, True),
    ):
        result = facet(scene, 1) if needs_grid else facet(scene)
        assert result.state is ResultState.VALUE, facet.__name__
        for key, value in result.subterms.items():
            # The exact path must keep publishing plain floats (no leaked graph).
            assert isinstance(value, float), key
            exact[key] = value
    for row, expected in exact.items():
        assert row in traced.traced_subterms, row
        traced_value = float(traced.traced_subterms[row].detach())
        # Traced forwards may differ from exact by accumulation order only.
        assert traced_value == pytest.approx(expected, rel=1e-9, abs=1e-15), row
