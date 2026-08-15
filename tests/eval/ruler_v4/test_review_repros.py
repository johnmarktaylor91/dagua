"""Live repros from the P4 adversarial review rounds, banked as regressions.

Every test here is a repro that a review lane (P4REVIEW / P4REVERIFY /
P4REVERIFY2 / P4REVERIFY3, Fable and Opus) ran live against a shipped
defect. They are permanent: each one pins the contract-lawful outcome the
fix round established, so no later pass can silently re-open the channel.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pytest
import torch

from dagua.eval.ruler_v4.ingestion import ingest
from dagua.eval.ruler_v4.legibility import U20a
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


def _node_scene(
    positions: torch.Tensor,
    ranks: Optional[Tuple[int, ...]] = None,
    flow_axis: Optional[Tuple[float, float]] = None,
    edges: Tuple[Tuple[int, int], ...] = (),
) -> Scene:
    """Ingest one edge-light repro scene.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    ranks : tuple[int, ...] or None
        Declared ranks, if any.
    flow_axis : tuple[float, float] or None
        Declared flow axis, if any.
    edges : tuple[tuple[int, int], ...]
        Canonical graph edges.

    Returns
    -------
    Scene
        Validated static scene.
    """

    graph = GraphSemantics(
        tuple(f"n{index}" for index in range(positions.shape[0])),
        edges,
        ranks=ranks,
        flow_axis=flow_axis,
    )
    routes = tuple(
        Route(index, torch.stack((positions[source], positions[target])))
        for index, (source, target) in enumerate(edges)
    )
    result = ingest(
        graph,
        DrawingScene(positions, routes),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes"})),
    )
    assert isinstance(result, ValidScene)
    return result.scene


# --- P4REVERIFY3_FABLE blocker 1: declaring ranks must not exempt a line ---
# collapse perpendicular to the declared axis (the E3-banned game channel).


def test_u20a_rank_declaration_cannot_exempt_cross_axis_line_collapse() -> None:
    """A horizontal line with vertical-axis ranks stays catastrophic."""

    positions = torch.tensor([[4.0 * index, 0.0] for index in range(6)], dtype=torch.float64)
    declared = U20a(_node_scene(positions, ranks=tuple(range(6)), flow_axis=(0.0, 1.0)))
    undeclared = U20a(_node_scene(positions))
    assert declared.subterms["U20a.ii"] == pytest.approx(1.0, abs=0.0)
    # Mutating the declared rank block cannot improve the composite (golden 3).
    assert declared.value == undeclared.value == pytest.approx(1.0, abs=0.0)


# --- P4REVERIFY3_FABLE blocker 2: the residual frame must not score every ---
# jittered multi-node-per-rank layered drawing a vacuous worst.


@pytest.mark.parametrize("eps", (0.0, 1e-6, 1e-3, 0.1, 0.5))
def test_u20a_jittered_layered_grid_is_not_rank_collapsed(eps: float) -> None:
    """E2 is exemption-only: a healthy raw quotient survives declared ranks."""

    rows = []
    for rank in range(3):
        rows.append([0.0, 8.0 * rank + eps])
        rows.append([6.0, 8.0 * rank - eps])
    positions = torch.tensor(rows, dtype=torch.float64)
    facet = U20a(_node_scene(positions, ranks=(0, 0, 1, 1, 2, 2), flow_axis=(0.0, 1.0)))
    assert facet.subterms["U20a.ii"] == pytest.approx(0.0, abs=0.0)


def test_u20a_exact_declared_axis_column_scores_zero() -> None:
    """The E2 exemption itself: a fully rank-explained column is expected."""

    positions = torch.tensor([[0.0, 4.0 * rank] for rank in range(6)], dtype=torch.float64)
    facet = U20a(_node_scene(positions, ranks=tuple(range(6)), flow_axis=(0.0, 1.0)))
    assert facet.subterms["U20a.ii"] == pytest.approx(0.0, abs=0.0)
    assert facet.value == pytest.approx(0.0, abs=0.0)


# --- P4REVERIFY3_FABLE blocker 3: a declared cluster drawn as separated ---
# lumps must ingest and be scored (and punished), never abort the scene.


def test_scattered_cluster_scene_ingests_and_every_facet_evaluates() -> None:
    """A two-lump cluster whose core center misses the region stays scorable."""

    from dagua.eval.ruler_v4 import FACET_FUNCTIONS, evaluate_facet

    points = [[float(index % 5), float(index // 5)] for index in range(15)]
    points += [[100.0 + float(index % 5), float(index // 5)] for index in range(15)]
    positions = torch.tensor(points, dtype=torch.float64)
    graph = GraphSemantics(
        tuple(f"n{index}" for index in range(30)),
        (),
        clusters={"lump": tuple(range(30))},
    )
    result = ingest(
        graph,
        DrawingScene(positions),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes", "cluster_labels"})),
    )
    assert isinstance(result, ValidScene)
    for facet_id in sorted(FACET_FUNCTIONS):
        evaluate_facet(facet_id, result.scene)
    punished = evaluate_facet("U30", result.scene, alpha_grid_index=1)
    assert punished.state is ResultState.VALUE
    assert punished.value is not None and punished.value > 0.0


# --- P4REVERIFY3_FABLE blocker 4: the global blend must stay total on ---
# saturated defect populations (float dust pushes an exact-1.0 trim above 1).


def test_u20a_scores_saturated_populations_instead_of_crashing() -> None:
    """Golden 2's catastrophic endpoint scores ~1; it must never raise."""

    coincident = U20a(_node_scene(torch.zeros((6, 2), dtype=torch.float64)))
    assert coincident.state is ResultState.VALUE
    assert coincident.value == pytest.approx(1.0, abs=1e-6)
    nano = U20a(
        _node_scene(
            torch.tensor(
                [[0.0, 0.0], [1e-9, 0.0], [1e-9, 1e-9], [0.0, 1e-9]],
                dtype=torch.float64,
            ),
            edges=((0, 1), (1, 2), (2, 3)),
        )
    )
    assert nano.state is ResultState.VALUE
    assert nano.value == pytest.approx(1.0, abs=1e-6)


def test_u11_scores_plain_declared_rank_dag() -> None:
    """A load-bearing corpus family must evaluate, not die in the blend."""

    from dagua.eval.ruler_v4.edges import U11

    points = [[4.0 * column, 6.0 * rank] for rank in range(3) for column in range(3)]
    edges = tuple(
        (rank * 3 + column, (rank + 1) * 3 + child)
        for rank in range(2)
        for column in range(3)
        for child in range(3)
        if (column + child) % 2 == 0
    )
    scene = _node_scene(
        torch.tensor(points, dtype=torch.float64),
        ranks=tuple(rank for rank in range(3) for _ in range(3)),
        flow_axis=(0.0, 1.0),
        edges=edges,
    )
    result = U11(scene)
    assert result.state is ResultState.VALUE
    assert result.subterms
