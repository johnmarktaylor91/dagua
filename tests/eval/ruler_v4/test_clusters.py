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


def test_u30_single_visible_cluster_label_pins_grid_envelope() -> None:
    """U30 pins the exact pre-selection envelope for one derived cluster label."""

    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    result = U30(
        _scene(
            positions,
            {"c": (0, 1, 2)},
            cluster_labels_visible=True,
        ),
        None,
    )
    assert result.state is ResultState.NA
    assert result.reason == "alpha_grid_unselected"
    lower, upper = result.raw["grid_envelope"]
    # U30 contract golden 4: "the occluded party pays; permuting z never repairs
    # a label overlap." Every preregistered alpha row therefore remains nonzero.
    assert 0.0 < lower <= upper <= 1.0
