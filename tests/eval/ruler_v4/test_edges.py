"""Worked examples for edge-geometry and routed-edge facet contracts."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Tuple

import torch

from dagua.eval.ruler_v4.edges import U07, U08, U10, U11, U12, U13, U15, U16
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
    edges: Tuple[Tuple[int, int], ...],
    routes: Optional[Tuple[Route, ...]] = None,
    graph_options: Optional[Mapping[str, Any]] = None,
    edge_label_positions: Optional[torch.Tensor] = None,
) -> Scene:
    """Ingest one routed worked-example scene.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edges : tuple[tuple[int, int], ...]
        Canonical graph edges.
    routes : tuple[Route, ...] or None
        Explicit routes, or straight routes when omitted.
    graph_options : mapping[str, Any] or None
        Optional GraphSemantics field overrides.
    edge_label_positions : torch.Tensor or None
        Optional label centers with shape ``[E, 2]``.

    Returns
    -------
    Scene
        Validated routed scene.
    """

    options = dict(graph_options or {})
    graph = GraphSemantics(
        tuple(f"n{index}" for index in range(positions.shape[0])), edges, **options
    )
    route_values = routes or tuple(
        Route(index, torch.stack((positions[source], positions[target])))
        for index, (source, target) in enumerate(edges)
    )
    profile = ObservationProfile(visible_channels=frozenset({"nodes", "routes", "edge_labels"}))
    result = ingest(
        graph,
        DrawingScene(
            positions,
            route_values,
            edge_label_positions=edge_label_positions,
        ),
        StyleContract(),
        profile,
    )
    assert isinstance(result, ValidScene)
    return result.scene


def test_u07_crossing_free_routes_have_exact_zero_defect() -> None:
    """U07's crossing-free golden returns exact zeros rather than NA."""

    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    result = U07(_scene(positions, ((0, 1), (1, 2))))
    assert result.state is ResultState.VALUE
    assert result.value == 0.0
    assert result.raw["crossing_count"] == 0


def test_u08_equal_six_spoke_star_has_exact_zero_defect() -> None:
    """U08's equal 60-degree spoke fixture has exact fair angular resolution."""

    angles = torch.arange(6, dtype=torch.float64) * (2.0 * math.pi / 6.0)
    leaves = 5.0 * torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    positions = torch.cat((torch.zeros((1, 2), dtype=torch.float64), leaves), dim=0)
    result = U08(_scene(positions, tuple((0, index) for index in range(1, 7))))
    assert result.state is ResultState.VALUE
    assert result.value is not None and result.value < 1e-15


def test_u10_clean_route_has_exact_zero_clearance_burden() -> None:
    """U10's clean route beyond the compact clearance band contributes zero."""

    positions = torch.tensor([[0.0, 0.0], [4.0, 0.0], [2.0, 5.0]])
    result = U10(_scene(positions, ((0, 1),)))
    assert result.state is ResultState.VALUE
    assert result.value == 0.0


def test_u11_straight_route_hits_all_five_anchored_zeros() -> None:
    """U11's straight zero-bend route has zero on every routed-quality row."""

    positions = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
    result = U11(_scene(positions, ((0, 1),)))
    assert result.state is ResultState.VALUE
    assert set(result.subterms.values()) == {0.0}


def test_u12_collinear_subdivided_path_has_zero_continuity_defect() -> None:
    """U12 is invariant to collinear route subdivision and returns exact zero."""

    positions = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
    route = Route(0, torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]))
    result = U12(_scene(positions, ((0, 1),), (route,)))
    assert result.state is ResultState.VALUE
    assert result.value == 0.0


def test_u13_well_separated_parallel_routes_hit_compact_zero() -> None:
    """U13's support-edge golden is exactly zero beyond 3.5 intrinsic units."""

    positions = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 7.0], [10.0, 7.0]])
    result = U13(_scene(positions, ((0, 1), (2, 3))))
    assert result.state is ResultState.VALUE
    assert result.value == 0.0


def test_u15_separated_parallel_arcs_have_zero_merge_defect() -> None:
    """U15's doubled-edge golden is zero when relative separation exceeds five percent."""

    positions = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    routes = (
        Route(0, torch.tensor([[0.0, 0.0], [2.0, 1.0], [8.0, 1.0], [10.0, 0.0]])),
        Route(1, torch.tensor([[0.0, 0.0], [2.0, -1.0], [8.0, -1.0], [10.0, 0.0]])),
    )
    result = U15(_scene(positions, ((0, 1), (0, 1)), routes))
    assert result.state is ResultState.VALUE
    assert result.subterms["U15.i"] == 0.0


def test_u16_clean_edge_label_has_zero_overlap() -> None:
    """U16's clean labeled path has no label-to-obstacle overlap."""

    positions = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    route = Route(0, torch.tensor([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]))
    result = U16(
        _scene(
            positions,
            ((0, 1),),
            (route,),
            {"edge_labels": ("edge",)},
            torch.tensor([[5.0, 1.5]]),
        )
    )
    assert result.state is ResultState.VALUE
    assert result.subterms["U16.i"] == 0.0
