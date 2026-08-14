"""Shared RULER V4 scene fixtures."""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

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


@pytest.fixture
def semantic_scene() -> Scene:
    """Build one directed, clustered, labelled scene.

    Returns
    -------
    Scene
        Validated eight-node fixture.
    """

    positions = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, -2.0],
            [2.0, 2.0],
            [4.0, -2.0],
            [4.0, 2.0],
            [6.0, -2.0],
            [6.0, 2.0],
            [8.0, 0.0],
        ],
        dtype=torch.float64,
    )
    edges: Tuple[Tuple[int, int], ...] = (
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    )
    graph = GraphSemantics(
        node_ids=tuple(f"n{index}" for index in range(8)),
        edges=edges,
        directed=True,
        node_labels=tuple(f"n{index}" for index in range(8)),
        edge_labels=tuple(None for _ in edges),
        clusters={"left": (0, 1, 2, 3), "right": (4, 5, 6, 7), "parent": tuple(range(8))},
        cluster_parents={"left": "parent", "right": "parent"},
        ranks=(0, 1, 1, 2, 2, 3, 3, 4),
        roots=(0,),
        feedback=tuple(False for _ in edges),
        edge_weights=tuple(float(index + 1) for index in range(len(edges))),
        weight_semantics="distance_cost",
        required_primitives=frozenset({"nodes", "routes"}),
    )
    routes = tuple(
        Route(index, torch.stack((positions[source], positions[target])))
        for index, (source, target) in enumerate(edges)
    )
    drawing = DrawingScene(positions, routes, ("nodes", "routes", "node_labels"))
    result = ingest(
        graph,
        drawing,
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes", "node_labels"})),
    )
    assert isinstance(result, ValidScene)
    return result.scene
