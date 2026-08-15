"""Worked examples for structural and frame-shape facets."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Optional, Tuple

import pytest
import torch

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
from dagua.eval.ruler_v4.structure import (
    U01,
    U02,
    U03,
    U05,
    U06,
    U09,
    U14,
    U22,
    U23,
    U24,
    U01b,
    U04a,
    U04b,
)


def _scene(
    positions: torch.Tensor,
    edges: Tuple[Tuple[int, int], ...],
    graph_options: Optional[Mapping[str, Any]] = None,
) -> Scene:
    """Ingest a compact worked-example scene.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edges : tuple[tuple[int, int], ...]
        Canonical graph edges.
    graph_options : mapping[str, Any] or None
        Optional GraphSemantics field overrides.

    Returns
    -------
    Scene
        Validated scene with one straight route per edge.
    """

    options = dict(graph_options or {})
    graph = GraphSemantics(
        node_ids=tuple(f"n{index}" for index in range(positions.shape[0])),
        edges=edges,
        **options,
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
    assert result.value == pytest.approx(0.0, abs=0.0)


def test_u01b_small_component_drops_empty_long_band() -> None:
    """U01b's sub-30-pair bands produce the frozen typed absence."""

    positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    result = U01b(_scene(positions, ((0, 1), (1, 2))))
    assert result.state is ResultState.NA
    assert result.reason == "band_underpopulated"
    assert result.raw["band_counts"] == {"local": 3, "long": 0}


def test_u01b_long_path_pins_both_shared_fit_band_values() -> None:
    """U01b's two populated bands read exact zero from one shared path fit."""

    positions = torch.stack((torch.arange(25, dtype=torch.float64), torch.zeros(25)), dim=1)
    result = U01b(_scene(positions, tuple((index, index + 1) for index in range(24))))
    assert result.state is ResultState.VALUE
    assert result.value == pytest.approx(0.0, abs=0.0)


def test_u02_monotone_path_has_perfect_rank_fidelity() -> None:
    """U02's perfect monotone path has rho one and exact zero defect."""

    positions = torch.stack((torch.arange(5, dtype=torch.float64), torch.zeros(5)), dim=1)
    result = U02(_scene(positions, tuple((index, index + 1) for index in range(4))))
    assert result.state is ResultState.VALUE
    assert result.value == pytest.approx(0.0, abs=0.0)


def test_u03_complete_graph_is_typed_saturated_absence() -> None:
    """U03's complete-graph golden is NA because every radius is saturated."""

    angles = torch.arange(6, dtype=torch.float64) * (2.0 * math.pi / 6.0)
    positions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    edges = tuple((left, right) for left in range(6) for right in range(left + 1, 6))
    result = U03(_scene(positions, edges))
    assert result.state is ResultState.NA
    assert result.reason == "neighborhoods_saturated"


def test_u03_hub_chain_pins_contiguous_degree_terciles() -> None:
    """U03's hub-and-chain golden hand-checks degree-stratum assignment."""

    positions = torch.stack((torch.arange(9, dtype=torch.float64), torch.zeros(9)), dim=1)
    edges = tuple((0, node) for node in range(1, 9)) + tuple(
        (node, node + 1) for node in range(1, 8)
    )
    result = U03(_scene(positions, edges))
    assert result.state is ResultState.VALUE
    # U03 golden 6 requires a hand-checked hub-and-chain tercile fixture.
    # Hand-checkable degree classes: chain ends {1, 8} have degree 2, interior
    # {2..7} degree 3, hub {0} degree 8. The six tied degree-3 nodes split by a
    # canonical tie key, so the classes are pinned, not the sample order.
    terciles = result.raw["degree_terciles"]["r_1.component_0"]
    assert tuple(len(tercile) for tercile in terciles) == (3, 3, 3)
    assert set(terciles[0]) | set(terciles[1]) | set(terciles[2]) == set(range(9))
    assert {1, 8} <= set(terciles[0])
    assert set(terciles[1]) <= set(range(2, 8))
    assert 0 in terciles[2]
    defects = result.raw["degree_stratum_defects"]["r_1.component_0"]
    assert max(defects) > min(defects)


def test_u04a_regular_cycle_has_identical_density_fields() -> None:
    """U04a pins the exact two-scale rotation-averaged cycle field divergence."""

    angles = torch.arange(10, dtype=torch.float64) * (2.0 * math.pi / 10.0)
    positions = 5.0 * torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    edges = tuple((index, (index + 1) % 10) for index in range(10))
    original = U04a(_scene(positions, edges))
    translated = U04a(_scene(positions + torch.tensor([17.0, -23.0]), edges))
    assert original.state is ResultState.VALUE
    assert translated.state is ResultState.VALUE
    # U04a section 11 freezes exact translation invariance through frame anchoring.
    assert translated.value == pytest.approx(original.value, abs=1e-12)


def test_u04b_generous_spacing_has_low_crowding() -> None:
    """U04b's sparse fixture has exactly zero above-knee coverage burden."""

    positions = torch.stack((5.0 * torch.arange(10, dtype=torch.float64), torch.zeros(10)), dim=1)
    result = U04b(_scene(positions, tuple((index, index + 1) for index in range(9))))
    assert result.state is ResultState.VALUE
    assert result.value == pytest.approx(0.0, abs=0.0)


def test_u05_path_matches_zero_stress_shape_golden() -> None:
    """U05's path resistance order is represented perfectly by a collinear path."""

    positions = torch.stack((torch.arange(20, dtype=torch.float64), torch.zeros(20)), dim=1)
    result = U05(_scene(positions, tuple((index, index + 1) for index in range(19))))
    assert result.state is ResultState.VALUE
    assert result.value == pytest.approx(0.0, abs=0.0)


def test_u06_regular_hexagon_realizes_certified_rotation() -> None:
    """U06's certified cycle generator has zero Procrustes residual on a hexagon."""

    angles = torch.arange(6, dtype=torch.float64) * (2.0 * math.pi / 6.0)
    positions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    edges = tuple((index, (index + 1) % 6) for index in range(6))
    generator = tuple((index + 1) % 6 for index in range(6))
    result = U06(_scene(positions, edges, {"symmetry_generators": (generator,)}))
    assert result.state is ResultState.VALUE
    assert result.value == pytest.approx(0.0, abs=1e-12)


def test_u09_unit_edge_lengths_have_zero_dispersion() -> None:
    """U09's unit-length path has exact zero median absolute dispersion."""

    positions = torch.stack((torch.arange(6, dtype=torch.float64), torch.zeros(6)), dim=1)
    result = U09(_scene(positions, tuple((index, index + 1) for index in range(5))))
    assert result.state is ResultState.VALUE
    assert result.value == pytest.approx(0.0, abs=0.0)


def test_u14_well_separated_nonedge_reaches_compact_zero() -> None:
    """U14's compact clearance kernel is exactly zero beyond one intrinsic unit."""

    positions = torch.stack((4.0 * torch.arange(8, dtype=torch.float64), torch.zeros(8)), dim=1)
    result = U14(_scene(positions, tuple((index, index + 1) for index in range(7))))
    assert result.state is ResultState.VALUE
    assert result.value == pytest.approx(0.0, abs=0.0)


def test_u22_wide_plateau_has_zero_cost(semantic_scene: Scene) -> None:
    """U22 charges no aspect within a factor three of its frozen target."""

    scene = replace(semantic_scene, graph=replace(semantic_scene.graph, ranks=None))
    result = U22(scene)
    assert result.state is ResultState.VALUE
    assert result.value == pytest.approx(0.0, abs=0.0)
    assert result.raw["measurement"] == "frozen_direction_set"


def test_u22_unknown_declared_class_is_typed_invalid() -> None:
    """U22 section 13 case (c): an unlisted class string never scores."""

    positions = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    result = U22(_scene(positions, (), {"declared_graph_class": "novel"}))
    # A silent kappa_class = 1 fallback is the branch section 13 pre-bans.
    assert result.state is ResultState.INVALID
    assert result.reason == "unknown_declared_class"


# --- U22 section 6 class-exemption table goldens (P4REVERIFY5 MINOR-1): ---
# every multiplier entry, in both orientations where a direction exists.
# kappa_class constants are elongation magnitudes; in the signed declared-axis
# frame each maps through its class's elongation direction (DISCREPANCIES.md
# entry 30).


def _soft_pos(argument: float) -> float:
    """Evaluate U22 section 6's frozen C^1 hinge."""

    return 0.0 if argument <= 0.0 else argument**2 / (argument + 0.5)


def _u22_closed_form(observed: float, target: float) -> float:
    """Evaluate D_U22 from section 6's frozen normalization."""

    excess = _soft_pos(abs(math.log(observed / target)) - math.log(3.0))
    return excess / (1.0 + excess)


def test_u22_declared_path_target_is_reciprocal_kappa_in_signed_frame() -> None:
    """path/chain kappa = 8 elongates along the axis: signed target 1/8."""

    along = torch.tensor([[0.0, 3.0 * index] for index in range(10)], dtype=torch.float64)
    across = torch.tensor([[3.0 * index, 0.0] for index in range(10)], dtype=torch.float64)
    for declared in ("path", "chain"):
        options = {"flow_axis": (0.0, 1.0), "declared_graph_class": declared}
        correct = U22(_scene(along, (), options))
        wrong = U22(_scene(across, (), options))
        assert correct.raw["target"] == pytest.approx(1.0 / 8.0, abs=0.0)
        # Drawing a declared path along its declared flow must beat drawing
        # it at 90 degrees (V4_SPEC 5.3 #17 mutation direction).
        assert correct.value < 0.02 < 0.75 < wrong.value
        assert correct.value == pytest.approx(
            _u22_closed_form(correct.raw["aspect_ratio"], 1.0 / 8.0), abs=1e-12
        )


def test_u22_declared_path_long_strip_scores_exact_zero_inside_plateau() -> None:
    """U22 section 15 golden 6: a declared path strip scores 0, undeclared > 0."""

    # 2.0u spacing keeps the strip's aspect inside the frozen kappa_class *
    # kappa_ar = 8 x 3 = 24 plateau, so the exemption's zero is exact
    # (P4REVERIFY6 MINOR-1; the 3.0u fixture above sits past the plateau and
    # pins the nonzero regression signature instead).
    strip = torch.tensor([[0.0, 2.0 * index] for index in range(10)], dtype=torch.float64)
    declared = U22(_scene(strip, (), {"flow_axis": (0.0, 1.0), "declared_graph_class": "path"}))
    undeclared = U22(_scene(strip, (), {"flow_axis": (0.0, 1.0)}))
    assert declared.value == pytest.approx(0.0, abs=0.0)
    assert declared.raw["exemption"] == "class:path"
    assert undeclared.value is not None and undeclared.value > 0.5
    assert undeclared.raw["exemption"] == "unit"


def test_u22_declared_tree_breadth_fold_passes_through_signed_frame() -> None:
    """tree max(1, b/d) is already breadth-over-depth: wide is expected."""

    wide = torch.tensor(
        [[0.0, 0.0], [-4.5, 3.0], [-1.5, 3.0], [1.5, 3.0], [4.5, 3.0]], dtype=torch.float64
    )
    deep = torch.tensor(
        [[0.0, 0.0], [3.0, -4.5], [3.0, -1.5], [3.0, 1.5], [3.0, 4.5]], dtype=torch.float64
    )
    options = {
        "flow_axis": (0.0, 1.0),
        "declared_graph_class": "tree",
        "tree_depths": (0, 2, 2, 2, 2),
    }
    correct = U22(_scene(wide, (), options))
    wrong = U22(_scene(deep, (), options))
    # b = 4 leaves on one level, depth d = 2: kappa = max(1, b/d) = 2.
    assert correct.raw["target"] == pytest.approx(2.0, abs=0.0)
    assert correct.value == pytest.approx(0.0, abs=0.0)
    assert wrong.value is not None and wrong.value > 0.4


def test_u22_declared_grid_aspect_is_signed_and_orientation_sensitive() -> None:
    """grid dims are width/height in the signed frame; mis-declaring pays."""

    positions = torch.tensor(
        [[3.0 * column, 3.0 * row] for row in range(2) for column in range(4)],
        dtype=torch.float64,
    )
    correct = U22(
        _scene(
            positions,
            (),
            {
                "flow_axis": (0.0, 1.0),
                "declared_graph_class": "grid",
                "lattice_dimensions": (4, 2),
            },
        )
    )
    mis_declared = U22(
        _scene(
            positions,
            (),
            {
                "flow_axis": (0.0, 1.0),
                "declared_graph_class": "grid",
                "lattice_dimensions": (2, 4),
            },
        )
    )
    # A 4x3u-by-1x3u lattice of unit-mass points has exact aspect 2.
    assert correct.raw["aspect_ratio"] == pytest.approx(2.0, abs=1e-12)
    assert correct.value == pytest.approx(0.0, abs=0.0)
    # Mis-declared dims: x = |log(2 / 0.5)|, through the frozen hinge.
    assert mis_declared.value == pytest.approx(_u22_closed_form(2.0, 0.5), abs=1e-12)
    assert mis_declared.value == pytest.approx(0.095079, abs=1e-6)


def test_u22_direction_free_grid_target_folds_onto_elongation_side() -> None:
    """Without an axis the sub-unit declared aspect folds to >= 1 (entry 26)."""

    positions = torch.tensor(
        [[3.0 * column, 3.0 * row] for row in range(4) for column in range(2)],
        dtype=torch.float64,
    )
    facet = U22(
        _scene(
            positions,
            (),
            {"declared_graph_class": "grid", "lattice_dimensions": (2, 4)},
        )
    )
    assert facet.raw["measurement"] == "frozen_direction_set"
    assert facet.raw["target"] == pytest.approx(2.0, abs=0.0)
    assert facet.value == pytest.approx(0.0, abs=0.0)


def test_u22_declared_ring_keeps_unit_target() -> None:
    """cycle/ring kappa = 1 is convention-free; a round ring is at target."""

    angles = torch.arange(8, dtype=torch.float64) * (2.0 * math.pi / 8.0)
    positions = 6.0 * torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    for declared in ("cycle", "ring"):
        facet = U22(
            _scene(positions, (), {"flow_axis": (0.0, 1.0), "declared_graph_class": declared})
        )
        assert facet.raw["target"] == pytest.approx(1.0, abs=0.0)
        assert facet.value == pytest.approx(0.0, abs=0.0)


def test_u23_symmetric_fixture_is_balanced(semantic_scene: Scene) -> None:
    """U23 returns exact zero for a centered symmetric drawing."""

    result = U23(semantic_scene)
    assert result.state is ResultState.VALUE
    assert result.value == pytest.approx(0.0, abs=0.0)


def test_u23_does_not_fabricate_axis_from_ranks() -> None:
    """U23 uses rotation averaging when ranks have no declared world axis."""

    positions = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    edges = ((0, 1), (1, 3), (3, 2), (2, 0))
    result = U23(_scene(positions, edges, {"ranks": (0, 0, 1, 1)}))
    # U23 section 5 requires a declared cross-axis, never one inferred from ranks.
    assert result.raw["measurement"] == "rotation_averaged"


def test_u24_compact_scene_publishes_finite_ink_ratio() -> None:
    """U24's compact node-only fixture produces a finite bounded ink defect."""

    positions = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    result = U24(_scene(positions, ((0, 1), (1, 3), (3, 2), (2, 0))))
    assert result.state is ResultState.VALUE
    assert result.value == pytest.approx(0.0, abs=0.0)
    assert result.raw["ink_ratio"] > 0.0
