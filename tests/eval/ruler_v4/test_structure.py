"""Worked examples for structural and frame-shape facets."""

from __future__ import annotations

from dataclasses import replace

import torch

from dagua.eval.ruler_v4.scene import ResultState, Scene
from dagua.eval.ruler_v4.structure import U01, U22, U23


def test_u01_path_with_unit_spacing_has_zero_stress(semantic_scene: Scene) -> None:
    """U01 required path golden has exact monotone stress zero."""

    positions = torch.stack(
        (torch.arange(semantic_scene.node_count, dtype=torch.float64), torch.zeros(8)), dim=1
    )
    edges = tuple((index, index + 1) for index in range(7))
    graph = replace(semantic_scene.graph, edges=edges, directed=False, ranks=None)
    scene = replace(semantic_scene, graph=graph, positions=positions, routes=())
    result = U01(scene)
    assert result.state is ResultState.VALUE
    assert result.value == 0.0


def test_u22_wide_plateau_has_zero_cost(semantic_scene: Scene) -> None:
    """U22 charges no aspect within a factor three of its frozen target."""

    scene = replace(semantic_scene, graph=replace(semantic_scene.graph, ranks=None))
    result = U22(scene)
    assert result.state is ResultState.VALUE
    assert result.value == 0.0
    assert result.raw["measurement"] == "frozen_direction_set"


def test_u23_symmetric_fixture_is_balanced(semantic_scene: Scene) -> None:
    """U23 returns exact zero for a centered symmetric drawing."""

    result = U23(semantic_scene)
    assert result.state is ResultState.VALUE
    assert result.value == 0.0
