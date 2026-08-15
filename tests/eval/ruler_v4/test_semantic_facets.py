"""Worked examples for directed, weighted, packing, and diagnostic contracts."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Tuple

import pytest
import torch

from dagua.eval.ruler_v4.directed import U31, U32, U33, U34, U39, _u34_blend
from dagua.eval.ruler_v4.ingestion import ingest, ingest_temporal
from dagua.eval.ruler_v4.packing import U38, U41, U42, _ciede2000
from dagua.eval.ruler_v4.registry import evaluate_facet
from dagua.eval.ruler_v4.scene import (
    ChannelDeclaration,
    DrawingScene,
    GraphSemantics,
    ObservationProfile,
    PortDeclaration,
    ResultState,
    Route,
    Scene,
    StyleContract,
    TemporalTransition,
    ValidScene,
    ValidTemporalScene,
)
from dagua.eval.ruler_v4.weights import U35, U36, U37


def _scene(
    positions: torch.Tensor,
    edges: Tuple[Tuple[int, int], ...],
    graph_options: Optional[Mapping[str, Any]] = None,
    style: Optional[StyleContract] = None,
) -> Scene:
    """Ingest one semantic worked-example scene.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edges : tuple[tuple[int, int], ...]
        Canonical graph edges.
    graph_options : mapping[str, Any] or None
        Optional GraphSemantics field overrides.
    style : StyleContract or None
        Optional corpus-owned style contract.

    Returns
    -------
    Scene
        Validated routed scene.
    """

    graph = GraphSemantics(
        tuple(f"n{index}" for index in range(positions.shape[0])),
        edges,
        **dict(graph_options or {}),
    )
    routes = tuple(
        Route(index, torch.stack((positions[source], positions[target])))
        for index, (source, target) in enumerate(edges)
    )
    result = ingest(
        graph,
        DrawingScene(positions, routes),
        style or StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes"})),
    )
    assert isinstance(result, ValidScene)
    return result.scene


def test_u31_axis_aligned_edges_have_exact_zero_direction_debt() -> None:
    """U31's aligned flow-axis fixture has exact zero signed direction loss."""

    positions = torch.tensor([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    scene = _scene(
        positions,
        ((0, 1), (1, 2)),
        {"directed": True, "flow_axis": (0.0, 1.0)},
    )
    result = U31(scene)
    assert result.state is ResultState.VALUE
    expected = 1.0 / (1.0 + math.exp(-((math.cos(math.radians(20.0)) - 1.0) / 0.01)))
    # U31 contract golden 1: aligned LR/TB fixtures have "formula-exact losses."
    assert result.value == pytest.approx(expected, abs=1e-15)


def test_u32_reversed_layers_are_worse_than_perfect_layers() -> None:
    """U32's reversed-layer fixture strictly worsens the isotonic subterm."""

    edges = ((0, 2), (1, 3))
    options = {"directed": True, "flow_axis": (0.0, 1.0), "ranks": (0, 0, 1, 1)}
    perfect = _scene(
        torch.tensor([[-1.0, 0.0], [1.0, 0.0], [-1.0, 4.0], [1.0, 4.0]]),
        edges,
        options,
    )
    reversed_scene = _scene(
        torch.tensor([[-1.0, 4.0], [1.0, 4.0], [-1.0, 0.0], [1.0, 0.0]]),
        edges,
        options,
    )
    perfect_result = U32(perfect)
    reversed_result = U32(reversed_scene)
    # U32 contract golden 1: reversed layers are asserted "RELATIONALLY" against
    # every correctly ordered fixture; opaque recomputed scalar pins are forbidden.
    assert perfect_result.value < 1e-12
    # U32 contract golden 1: reversed layers must be "STRICTLY WORSE" than the
    # corresponding correctly ordered fixture.
    assert reversed_result.value > perfect_result.value
    assert reversed_result.subterms["U32.L_iso"] > perfect_result.subterms["U32.L_iso"]


def test_u33_reversed_tree_depth_is_worse_than_layered_tree() -> None:
    """U33's layered tree fixture penalizes reversed parent-child progress."""

    edges = ((0, 1), (0, 2))
    options = {
        "directed": True,
        "flow_axis": (0.0, 1.0),
        "roots": (0,),
        "ranks": (0, 1, 1),
        "tree_parents": (None, 0, 0),
        "tree_depths": (0, 1, 1),
        "tree_layout": "layered",
        "ordered_children": {0: (1, 2)},
    }
    good = _scene(torch.tensor([[0.0, 0.0], [-2.0, 3.0], [2.0, 3.0]]), edges, options)
    bad = _scene(torch.tensor([[0.0, 3.0], [-2.0, 0.0], [2.0, 0.0]]), edges, options)
    good_result = U33(good)
    bad_result = U33(bad)
    # U33 contract golden 1: "Perfect, reversed-depth ... fixtures" pin the
    # direction; the perfect fixture has only the smooth logistic residue.
    assert good_result.value < 1e-6
    # U33 contract golden 1: the reversed-depth fixture is the relational bad case.
    assert bad_result.value > good_result.value
    assert bad_result.subterms["U33.layered.3"] > good_result.subterms["U33.layered.3"]


def test_u33_parent_centering_uses_declared_child_mass() -> None:
    """U33 centers a parent on the node-mass centroid of its children."""

    positions = torch.tensor([[1.0, 0.0], [0.0, 3.0], [10.0, 3.0]], dtype=torch.float64)
    options = {
        "directed": True,
        "flow_axis": (0.0, 1.0),
        "roots": (0,),
        "ranks": (0, 1, 1),
        "node_masses": (1.0, 9.0, 1.0),
        "tree_parents": (None, 0, 0),
        "tree_depths": (0, 1, 1),
        "tree_layout": "layered",
    }
    result = U33(_scene(positions, ((0, 1), (0, 2)), options))
    # U33 layered subterm 2 defines c as the node-mass child centroid.
    assert result.subterms["U33.layered.2"] == pytest.approx(0.0, abs=0.0)


def test_u34_straight_monotone_path_has_exact_zero_trace_debt() -> None:
    """U34's straight monotone source-to-sink path has zero on all rows."""

    positions = torch.tensor([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    scene = _scene(
        positions,
        ((0, 1), (1, 2)),
        {"directed": True, "flow_axis": (0.0, 1.0)},
    )
    result = U34(scene)
    assert result.state is ResultState.VALUE
    assert result.subterms["U34.L_back"] == pytest.approx(3.7200759760208366e-46, rel=1e-12)
    assert result.subterms["U34.L_mono"] == pytest.approx(3.3382377953649878e-15, rel=1e-12)
    assert result.subterms["U34.L_cont"] == pytest.approx(0.0, abs=0.0)


def test_u34_applies_frozen_source_sample_cap() -> None:
    """Use the contract's bottom-hash source panel above 64 sources."""

    source_count = 65
    positions = torch.tensor(
        [[float(pair), float(level)] for pair in range(source_count) for level in (0, 1)],
        dtype=torch.float64,
    )
    edges = tuple((2 * index, 2 * index + 1) for index in range(source_count))
    result = U34(
        _scene(
            positions,
            edges,
            {"directed": True, "flow_axis": (0.0, 1.0)},
        )
    )
    # U34 contract golden 2: "two-stage fixed-band bottom-hash selection";
    # section 1 caps the source panel at the 64 lowest hashes.
    assert result.raw["path_count"] == 64


def test_u34_uses_equal_stratum_ht_mean_component() -> None:
    """U34's dominant mean component equal-weights nonempty hop bands."""

    defects = [0.0, 1.0, 1.0]
    weights = [100.0, 1.0, 1.0]
    cvar = (2.0 / 102.0) / 0.10
    smooth_max = 1.0 + 0.05 * math.log((100.0 * math.exp(-20.0) + 2.0) / 102.0)
    expected = 0.65 * 0.5 + 0.25 * cvar + 0.10 * smooth_max
    # U34 section 4 replaces the robust mean with the equal-stratum HT mean.
    assert _u34_blend(defects, weights, [0, 1, 1]) == pytest.approx(expected, abs=1e-15)


def test_u35_constant_weights_match_unweighted_path_golden() -> None:
    """U35's constant-weight limit has exact zero on a perfect collinear path."""

    positions = torch.stack((torch.arange(5, dtype=torch.float64), torch.zeros(5)), dim=1)
    edges = tuple((index, index + 1) for index in range(4))
    scene = _scene(
        positions,
        edges,
        {
            "edge_weights": (1.0, 1.0, 1.0, 1.0),
            "weight_semantics": "distance_cost",
        },
    )
    result = U35(scene)
    assert result.state is ResultState.VALUE
    assert result.value == pytest.approx(0.0, abs=0.0)
    assert result.raw["alpha"] == 0.0


def test_u36_reversed_local_weight_order_is_worse() -> None:
    """U36's three-edge star prefers stronger edges drawn shorter."""

    edges = ((0, 1), (0, 2), (0, 3))
    options = {
        "edge_weights": (3.0, 2.0, 1.0),
        "weight_semantics": "connection_strength",
    }
    good = _scene(torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]]), edges, options)
    bad = _scene(torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 2.0], [-1.0, 0.0]]), edges, options)
    good_result = U36(good)
    bad_result = U36(bad)
    assert good_result.value is not None
    assert bad_result.value is not None
    expected_good = (
        sum(1.0 / (1.0 + math.exp(-(value / 0.03))) for value in (-1.0 / 3.0, -0.5, -0.2)) / 3.0
    )
    expected_bad = (
        sum(1.0 / (1.0 + math.exp(-(value / 0.03))) for value in (0.2, 0.5, 1.0 / 3.0)) / 3.0
    )
    # U36 golden 1 freezes ell=sigmoid(((l_strong-l_weak)/(l_strong+l_weak))/0.03).
    assert good_result.value == pytest.approx(expected_good, abs=1e-18)
    assert bad_result.value == pytest.approx(expected_bad, abs=1e-15)
    assert bad_result.value > good_result.value


def test_u37_exact_log_linear_width_map_has_near_zero_diagnostic() -> None:
    """U37's exact monotone encoding map agrees at every declared knot."""

    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
    edges = ((0, 1), (1, 2), (2, 3))
    scene = _scene(
        positions,
        edges,
        {
            "edge_weights": (1.0, 2.0, 4.0),
            "weight_semantics": "connection_strength",
            "weight_visual_channel": "stroke_thickness",
            "weight_encoding_knots": ((1.0, 1.0), (2.0, 2.0), (4.0, 4.0)),
        },
        StyleContract(edge_stroke_widths=(1.0, 2.0, 4.0)),
    )
    result = U37(scene)
    assert result.state is ResultState.VALUE
    assert result.value == pytest.approx(0.0, abs=1e-12)


def test_u38_wide_components_reduce_clearance_debt() -> None:
    """U38's two-component clearance row improves from overlap to wide separation."""

    edges = ((0, 1), (2, 3))
    overlap = _scene(torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.2, 0.0], [0.2, 1.0]]), edges)
    wide = _scene(torch.tensor([[0.0, 0.0], [0.0, 1.0], [8.0, 0.0], [8.0, 1.0]]), edges)
    overlap_result = U38(overlap)
    wide_result = U38(wide)
    # U38 contract golden 1: "overlap, contact, 0.5u, and wide clearance" must
    # follow the smooth signed-loss ordering.
    assert overlap_result.value > wide_result.value
    # U38 contract section 2 fixes every scored loss in [0,1].
    assert wide_result.value >= 0.0
    assert wide_result.subterms["U38.L_clear"] < overlap_result.subterms["U38.L_clear"]


def test_u39_exact_east_west_ports_have_zero_compliance_debt() -> None:
    """U39's exact anchors and approach directions have zero endpoint rows."""

    positions = torch.tensor([[0.0, 0.0], [4.0, 0.0]], dtype=torch.float64)
    source_port = PortDeclaration("p0", 0, "E", 0.5, 0, (1.0, 0.0), (1.0, 0.0))
    target_port = PortDeclaration("p1", 1, "W", 0.5, 0, (-1.0, 0.0), (-1.0, 0.0))
    graph = GraphSemantics(("n0", "n1"), ((0, 1),), ports={0: (source_port, target_port)})
    route = Route(
        0,
        torch.tensor([[0.51, 0.0], [3.49, 0.0]], dtype=torch.float64),
    )
    ingested = ingest(
        graph,
        DrawingScene(positions, (route,)),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes"})),
    )
    assert isinstance(ingested, ValidScene)
    scene = ingested.scene
    result = U39(scene)
    assert result.state is ResultState.VALUE
    assert result.subterms["U39.1"] == pytest.approx(0.0, abs=0.0)
    expected_approach = 1.0 / (1.0 + math.exp(-((math.cos(math.radians(25.0)) - 1.0) / 0.02)))
    # U39 contract golden 2: "correct/reversed approach" uses the formula-exact
    # 25-degree logistic shoulder for a correct east/west tangent.
    assert result.subterms["U39.2"] == pytest.approx(expected_approach, rel=1e-12)


def test_u40_identical_frames_have_zero_temporal_headline() -> None:
    """U40's identical-frame golden has zero rigidly aligned displacement and churn."""

    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    options = {"temporal_ids": ("a", "b", "c")}
    before = _scene(positions, ((0, 1), (0, 2)), options)
    after = _scene(positions.clone(), ((0, 1), (0, 2)), options)
    transition = TemporalTransition(
        {"a": "unchanged", "b": "unchanged", "c": "unchanged"},
        {"a": 0.0, "b": 0.0, "c": 0.0},
    )
    ingested = ingest_temporal((before, after), (transition,))
    assert isinstance(ingested, ValidTemporalScene)
    result = evaluate_facet("U40", ingested.scene)
    assert result.state is ResultState.VALUE
    assert result.value is None
    assert result.temporal_headline == pytest.approx(0.0, abs=0.0)
    assert result.raw["temporal_headline"] == result.temporal_headline


def test_u41_certified_triangle_pins_convexity_and_area_balance() -> None:
    """U41's certified triangle is perfectly convex and pays only area balance.

    The sole face equals its convex hull, so ``L_conv`` is sigmoid tails; the
    facet value is NOT zero because area balance is deliberately
    scale-sensitive (U41.md sec 4): the unit-scale face is far smaller than
    ``a_ref = A_ref / F0``. Both sub-terms and the value are pinned to the
    contract closed forms from fixture geometry and ingestion-published
    primitives, independently of the facet's own outputs.
    """

    height = 3.0**0.5
    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, height]], dtype=torch.float64)
    scene = _scene(
        positions,
        ((0, 1), (1, 2), (2, 0)),
        {"planarity_certificate": {"planar": True}},
    )
    result = U41(scene)
    assert result.state is ResultState.VALUE
    # U41 golden 1 names the triangle: its sole face equals its convex hull.
    assert result.raw["F0"] == 1
    assert result.raw["arrangement_face_count"] == 1
    assert result.raw["faces"][0]["convexity_defect"] == pytest.approx(0.0, abs=0.0)
    assert result.subterms["U41.L_conv"] == pytest.approx(0.0, abs=1e-12)
    # Contract quantities (U41.md secs 4, 6; U21.md A_ref with phi_target
    # = 0.10 and one component): a_f is the fixture triangle's closed-form
    # area, a_ref derives from the ingestion-published node boxes.
    face_area = math.sqrt(3.0)
    primitive_area = sum(
        float(4.0 * box.half_extents[0] * box.half_extents[1]) for box in scene.node_boxes
    )
    area_reference = primitive_area / 0.10
    assert result.raw["a_ref"] == pytest.approx(area_reference, rel=1e-12)
    balance_burden = abs(face_area - area_reference) / (face_area + area_reference)
    expected_area_loss = 1.0 - math.exp(-balance_burden)
    assert result.subterms["U41.L_area"] == pytest.approx(expected_area_loss, rel=1e-12)
    # Frozen 0.60/0.40 mass (U41.md sec 7) over independently derived terms.
    assert result.value == pytest.approx(0.40 * expected_area_loss, abs=1e-12)


def test_u42_default_v4_style_has_typed_channel_absence() -> None:
    """U42's v4.0 applicability fixture is NA without a declared channel set."""

    scene = _scene(torch.tensor([[0.0, 0.0], [2.0, 0.0]]), ((0, 1),))
    result = U42(scene)
    assert result.state is ResultState.NA
    assert result.reason == "no_declared_channels"


def test_u42_black_white_contrast_and_ciede2000_goldens() -> None:
    """U42 pins WCAG endpoints and the first Sharma CIEDE2000 vector."""

    declaration = ChannelDeclaration(
        "class_fill",
        "node",
        "class",
        "fill_color",
        {"black": (0.0, 0.0, 0.0), "white": (1.0, 1.0, 1.0)},
    )
    graph = GraphSemantics(
        ("n0", "n1"),
        (),
        node_attributes={"class": ("black", "white")},
        legends={"class_fill": declaration.value_map},
    )
    ingested = ingest(
        graph,
        DrawingScene(torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float64)),
        StyleContract(channel_set=(declaration,)),
        ObservationProfile(visible_channels=frozenset({"nodes"})),
    )
    assert isinstance(ingested, ValidScene)
    result = U42(ingested.scene)
    smooth_maximum = 1.0 + 0.05 * math.log((1.0 + math.exp(-20.0)) / 2.0)
    expected_contrast = 0.65 * 0.5 + 0.25 + 0.10 * smooth_maximum
    assert result.subterms["U42.i"] == pytest.approx(expected_contrast, abs=1e-15)
    assert _ciede2000(
        (50.0, 2.6772, -79.7751),
        (50.0, 0.0, -82.7485),
    ) == pytest.approx(2.0424596801565764, abs=1e-12)
