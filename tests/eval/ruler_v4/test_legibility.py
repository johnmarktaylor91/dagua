"""Worked examples for primitive legibility and frame-economy contracts."""

from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch

from dagua.eval.ruler_v4.ingestion import ingest
from dagua.eval.ruler_v4.legibility import U17, U18, U19, U21, U20a, U20b
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
    edges: Tuple[Tuple[int, int], ...] = (),
    labelled: bool = False,
) -> Scene:
    """Ingest one legibility worked-example scene.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edges : tuple[tuple[int, int], ...]
        Canonical graph edges.
    labelled : bool
        Whether every node has a visible declared label.

    Returns
    -------
    Scene
        Validated static scene.
    """

    labels = tuple(f"n{index}" for index in range(positions.shape[0])) if labelled else ()
    graph = GraphSemantics(
        tuple(f"n{index}" for index in range(positions.shape[0])),
        edges,
        node_labels=labels,
    )
    routes = tuple(
        Route(index, torch.stack((positions[source], positions[target])))
        for index, (source, target) in enumerate(edges)
    )
    channels = {"nodes", "routes"}
    if labelled:
        channels.add("node_labels")
    result = ingest(
        graph,
        DrawingScene(positions, routes),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset(channels)),
    )
    assert isinstance(result, ValidScene)
    return result.scene


def test_u17_generously_separated_nodes_have_exact_zero_defect() -> None:
    """U17's clear pair lies beyond the compact half-unit clearance support."""

    result = U17(_scene(torch.tensor([[0.0, 0.0], [5.0, 0.0]])), None)
    assert result.state is ResultState.NA
    assert result.reason == "alpha_grid_unselected"
    assert result.raw["grid_envelope"] == pytest.approx((0.0, 0.0), abs=0.0)


def test_u17_explicit_grid_row_returns_its_exact_scalar() -> None:
    """An explicit shared-grid row selects a scalar without inventing a default row."""

    scene = _scene(torch.tensor([[0.0, 0.0], [5.0, 0.0]]))
    selected = U17(scene, 1)
    assert selected.state is ResultState.VALUE
    assert selected.raw["alpha_grid_index"] == 1
    assert selected.raw["alpha_grid_name"] == "AC15_AH00"
    assert selected.value == pytest.approx(0.0, abs=0.0)


@pytest.mark.parametrize("alpha_grid_index", (0, 13, True))
def test_u17_rejects_non_grid_parameters(alpha_grid_index: int) -> None:
    """Reject off-list shared-grid parameters.

    Parameters
    ----------
    alpha_grid_index : int
        Invalid row supplied by pytest.
    """

    scene = _scene(torch.tensor([[0.0, 0.0], [5.0, 0.0]]))
    with pytest.raises(ValueError):
        U17(scene, alpha_grid_index)


def test_u18_separated_node_labels_have_zero_overlap_terms() -> None:
    """U18's separated declared labels have no label, node, or route collision."""

    result = U18(_scene(torch.tensor([[0.0, 0.0], [6.0, 0.0]]), labelled=True), None)
    assert result.state is ResultState.NA
    assert result.reason == "alpha_grid_unselected"
    assert result.raw["grid_envelope"] == pytest.approx((0.0, 0.0), abs=0.0)


def test_u19_v4_profile_without_physical_output_is_typed_na() -> None:
    """U19's frozen v4.0 applicability golden is NA without physical size."""

    result = U19(_scene(torch.tensor([[0.0, 0.0], [6.0, 0.0]]), labelled=True))
    assert result.state is ResultState.NA
    assert result.reason == "no_declared_physical_size"


def test_u19_physical_label_fixture_pins_exact_legibility_loss() -> None:
    """U19 pins the declared physical scale and quintic legibility loss."""

    positions = torch.tensor([[0.0, 0.0], [4.0, 0.0]], dtype=torch.float64)
    graph = GraphSemantics(("n0", "n1"), ((0, 1),), node_labels=("a", "b"))
    style = StyleContract(
        physical_output={
            "output_width": 10.0,
            "output_height": 10.0,
            "h_font": 1.0,
            "h_floor": 2.0,
        }
    )
    result = ingest(
        graph,
        DrawingScene(positions),
        style,
        ObservationProfile(visible_channels=frozenset({"nodes", "routes", "node_labels"})),
    )
    assert isinstance(result, ValidScene)
    facet = U19(result.scene)
    assert facet.state is ResultState.VALUE
    # U21's small-N frame has x half-extent 6 here, so the 10-unit viewport
    # gives m_phys=10/12 and r_l=(1)*(10/12)/2=5/12.
    ratio = 5.0 / 12.0
    expected = 1.0 - ratio**3 * (ratio * (6.0 * ratio - 15.0) + 10.0)
    assert 0.0 < facet.value < 1.0
    # U19 contract golden 3: "a label at half the floor scores in (0,1)."
    assert facet.value == pytest.approx(expected, abs=1e-15)


def test_u20a_declared_rank_column_uses_residual_frame() -> None:
    """E2 removes rank-explained axis variance from a perfect layered column."""

    positions = torch.tensor([[0.0, 4.0 * rank] for rank in range(6)], dtype=torch.float64)
    graph = GraphSemantics(
        tuple(f"n{rank}" for rank in range(6)),
        (),
        ranks=tuple(range(6)),
        flow_axis=(0.0, 1.0),
    )
    result = ingest(
        graph,
        DrawingScene(positions),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes"})),
    )
    assert isinstance(result, ValidScene)
    facet = U20a(result.scene)
    # U20a E2: correctly layered declared-axis collinearity is expected.
    assert facet.subterms["U20a.ii"] == pytest.approx(0.0, abs=0.0)
    assert facet.value == pytest.approx(0.0, abs=0.0)


def test_u20a_total_collapse_is_worse_than_two_dimensional_spread() -> None:
    """U20a's collapse fixture strictly worsens resolution-limit degeneracy."""

    spread = _scene(torch.tensor([[-4.0, -4.0], [-4.0, 4.0], [4.0, -4.0], [4.0, 4.0]]))
    collapsed = _scene(torch.zeros((4, 2), dtype=torch.float64))
    spread_result = U20a(spread)
    collapsed_result = U20a(collapsed)
    assert spread_result.value is not None
    assert collapsed_result.value is not None
    assert spread_result.value == pytest.approx(0.0, abs=0.0)
    assert collapsed_result.value == pytest.approx(1.0, abs=0.0)
    assert collapsed_result.value > spread_result.value


def test_u20a_exempts_incident_route_features() -> None:
    """Do not make every routed graph maximally degenerate at its terminals."""

    positions = torch.tensor([[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]])
    result = U20a(_scene(positions, ((0, 1), (1, 2), (2, 3), (3, 0))))
    assert result.state is ResultState.VALUE
    assert result.subterms["U20a.iii"] < 1.0
    assert result.value < 1.0


def test_u20a_scores_nonadjacent_segments_of_endpoint_sharing_routes() -> None:
    """U20a exempts only terminal-adjacent segments of routes sharing a node."""

    positions = torch.tensor([[0.0, 0.0], [10.0, 10.0], [10.0, 9.0]], dtype=torch.float64)
    graph = GraphSemantics(("n0", "n1", "n2"), ((0, 1), (0, 2)))
    routes = (
        Route(0, torch.tensor([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0]])),
        Route(1, torch.tensor([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [10.0, 9.0]])),
    )
    ingested = ingest(
        graph,
        DrawingScene(positions, routes),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes"})),
    )
    assert isinstance(ingested, ValidScene)
    # U20a section 3 admits non-adjacent segments even when routes share an endpoint.
    assert U20a(ingested.scene).subterms["U20a.iii"] > 0.0


def test_u20b_midscale_edges_lie_on_low_defect_plateau() -> None:
    """U20b's midscale edge fixture lies between short- and long-edge burdens."""

    positions = torch.tensor([[0.0, 0.0], [7.0, 0.0], [14.0, 0.0]])
    result = U20b(_scene(positions, ((0, 1), (1, 2))))
    assert result.state is ResultState.VALUE
    coordinate = math.log2(result.raw["median_edge_length_u"])
    expected = 1.0 / (1.0 + math.exp(-(math.log2(1.5) - coordinate) / 0.35))
    expected += 1.0 / (1.0 + math.exp(-(coordinate - math.log2(8.0)) / 0.35))
    # U20b contract golden 2: "monotone shoulders, flat plateau" under the
    # section-6 formula with anchors 1.5u and 8u and shoulder width 0.35.
    assert result.value == pytest.approx(expected, abs=1e-15)


def test_u21_compact_symmetric_scene_has_no_sparse_or_overflow_debt() -> None:
    """U21's compact symmetric fixture has exact zero frame-economy subterms."""

    positions = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    result = U21(_scene(positions, ((0, 1), (1, 3), (3, 2), (2, 0))))
    assert result.state is ResultState.VALUE
    assert result.value == pytest.approx(0.0, abs=0.0)


def test_u21_route_mass_is_invariant_to_declared_edge_weights() -> None:
    """U21 gives every escaped route unit mass regardless of weight semantics."""

    positions = torch.stack((torch.arange(30, dtype=torch.float64), torch.zeros(30)), dim=1)
    positions[-1] = torch.tensor([1000.0, 1000.0], dtype=torch.float64)
    edges = tuple((index, index + 1) for index in range(29))

    def weighted_scene(weights: Tuple[float, ...]) -> Scene:
        """Ingest one geometry with selected semantic edge weights.

        Parameters
        ----------
        weights : tuple[float, ...]
            Positive declared flow weights.

        Returns
        -------
        Scene
            Validated routed scene.
        """

        graph = GraphSemantics(
            tuple(f"n{index}" for index in range(30)),
            edges,
            edge_weights=weights,
            weight_semantics="flow",
        )
        routes = tuple(
            Route(index, torch.stack((positions[source], positions[target])))
            for index, (source, target) in enumerate(edges)
        )
        ingested = ingest(
            graph,
            DrawingScene(positions, routes),
            StyleContract(),
            ObservationProfile(visible_channels=frozenset({"nodes", "routes"})),
        )
        assert isinstance(ingested, ValidScene)
        return ingested.scene

    unit = U21(weighted_scene(tuple(1.0 for _ in edges)))
    skewed = U21(weighted_scene(tuple([1.0] * 28 + [100.0])))
    # U21 section 4 freezes route-ribbon mass at one per declared edge.
    assert skewed.raw["mass_out"] == pytest.approx(unit.raw["mass_out"], abs=0.0)
