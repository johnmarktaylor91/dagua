"""Worked examples for primitive legibility and frame-economy contracts."""

from __future__ import annotations

from typing import Tuple

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

    result = U17(_scene(torch.tensor([[0.0, 0.0], [5.0, 0.0]])))
    assert result.state is ResultState.VALUE
    assert result.value == 0.0


def test_u18_separated_node_labels_have_zero_overlap_terms() -> None:
    """U18's separated declared labels have no label, node, or route collision."""

    result = U18(_scene(torch.tensor([[0.0, 0.0], [6.0, 0.0]]), labelled=True))
    assert result.state is ResultState.VALUE
    assert set(result.subterms.values()) == {0.0}


def test_u19_v4_profile_without_physical_output_is_typed_na() -> None:
    """U19's frozen v4.0 applicability golden is NA without physical size."""

    result = U19(_scene(torch.tensor([[0.0, 0.0], [6.0, 0.0]]), labelled=True))
    assert result.state is ResultState.NA
    assert result.reason == "no_physical_viewport"


def test_u20a_total_collapse_is_worse_than_two_dimensional_spread() -> None:
    """U20a's collapse fixture strictly worsens resolution-limit degeneracy."""

    spread = _scene(torch.tensor([[-4.0, -4.0], [-4.0, 4.0], [4.0, -4.0], [4.0, 4.0]]))
    collapsed = _scene(torch.zeros((4, 2), dtype=torch.float64))
    spread_result = U20a(spread)
    collapsed_result = U20a(collapsed)
    assert spread_result.value is not None
    assert collapsed_result.value is not None
    assert collapsed_result.value > spread_result.value


def test_u20b_midscale_edges_lie_on_low_defect_plateau() -> None:
    """U20b's midscale edge fixture lies between short- and long-edge burdens."""

    positions = torch.tensor([[0.0, 0.0], [7.0, 0.0], [14.0, 0.0]])
    result = U20b(_scene(positions, ((0, 1), (1, 2))))
    assert result.state is ResultState.VALUE
    assert result.value is not None and result.value < 0.05


def test_u21_compact_symmetric_scene_has_no_sparse_or_overflow_debt() -> None:
    """U21's compact symmetric fixture has exact zero frame-economy subterms."""

    positions = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    result = U21(_scene(positions, ((0, 1), (1, 3), (3, 2), (2, 0))))
    assert result.state is ResultState.VALUE
    assert result.value == 0.0
