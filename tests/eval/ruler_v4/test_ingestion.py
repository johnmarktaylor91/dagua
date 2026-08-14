"""Three-way ingestion contract tests."""

from __future__ import annotations

import torch

from dagua.eval.ruler_v4.ingestion import ingest, ingest_record, ingest_temporal
from dagua.eval.ruler_v4.scene import (
    DrawingScene,
    GraphSemantics,
    IngestionErrorCode,
    InvalidScene,
    ObservationProfile,
    Route,
    StyleContract,
    TemporalTransition,
    ValidScene,
    ValidTemporalScene,
)


def _graph() -> GraphSemantics:
    """Return a minimal valid graph.

    Returns
    -------
    GraphSemantics
        Three-node path.
    """

    return GraphSemantics(("a", "b", "c"), ((0, 1), (1, 2)))


def test_valid_scene_derives_boxes_and_unit() -> None:
    """A valid producer scene derives all extents from StyleContract."""

    result = ingest(
        _graph(),
        DrawingScene(torch.tensor([[0, 0], [1, 0], [2, 0]])),
        StyleContract(),
        ObservationProfile(),
    )
    assert isinstance(result, ValidScene)
    assert len(result.scene.node_boxes) == 3
    assert result.scene.intrinsic_unit > 0.0
    assert result.scene.positions.dtype == torch.float64


def test_optional_absence_keeps_unaffected_scene_scoreable() -> None:
    """An optional missing channel does not discard the otherwise valid scene."""

    profile = ObservationProfile(optional_channels=frozenset({"node_labels"}))
    result = ingest(
        _graph(),
        DrawingScene(torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])),
        StyleContract(),
        profile,
    )
    assert isinstance(result, ValidScene)


def test_absent_simple_route_uses_render_truth_chord() -> None:
    """A missing simple-edge route remains valid under K17's chord identity."""

    graph = GraphSemantics(
        ("a", "b"), ((0, 1),), required_primitives=frozenset({"nodes", "routes"})
    )
    result = ingest(
        graph,
        DrawingScene(torch.tensor([[0.0, 0.0], [1.0, 0.0]])),
        StyleContract(),
        ObservationProfile(),
    )
    assert isinstance(result, ValidScene)
    assert result.scene.routes == ()


def test_nonfinite_geometry_is_invalid() -> None:
    """Malformed geometry becomes a typed error, never silent NaN."""

    result = ingest(
        _graph(),
        DrawingScene(torch.tensor([[0.0, 0.0], [float("nan"), 0.0], [2.0, 0.0]])),
        StyleContract(),
        ObservationProfile(),
    )
    assert isinstance(result, InvalidScene)
    assert result.code is IngestionErrorCode.NONFINITE_GEOMETRY


def test_malformed_position_shape_is_typed() -> None:
    """A rank-one position tensor returns MALFORMED_POSITIONS."""

    result = ingest(
        _graph(),
        DrawingScene(torch.tensor([0.0, 1.0, 2.0])),
        StyleContract(),
        ObservationProfile(),
    )
    assert isinstance(result, InvalidScene)
    assert result.code is IngestionErrorCode.MALFORMED_POSITIONS


def test_impossible_style_is_typed() -> None:
    """A nonpositive minimum extent returns IMPOSSIBLE_STYLE."""

    result = ingest(
        _graph(),
        DrawingScene(torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])),
        StyleContract(minimum_node_width=0.0),
        ObservationProfile(),
    )
    assert isinstance(result, InvalidScene)
    assert result.code is IngestionErrorCode.IMPOSSIBLE_STYLE


def test_unsupported_compositing_is_typed() -> None:
    """A non-opaque style returns UNSUPPORTED_COMPOSITING."""

    result = ingest(
        _graph(),
        DrawingScene(torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])),
        StyleContract(opaque=False),
        ObservationProfile(),
    )
    assert isinstance(result, InvalidScene)
    assert result.code is IngestionErrorCode.UNSUPPORTED_COMPOSITING


def test_malformed_route_shape_is_typed() -> None:
    """A one-point route returns MALFORMED_ROUTE."""

    route = Route(0, torch.tensor([[0.0, 0.0]]))
    result = ingest(
        _graph(),
        DrawingScene(
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            (route,),
        ),
        StyleContract(),
        ObservationProfile(),
    )
    assert isinstance(result, InvalidScene)
    assert result.code is IngestionErrorCode.MALFORMED_ROUTE


def test_detached_route_terminal_is_topology_mismatch() -> None:
    """A terminal farther than u/2 from its owner returns TOPOLOGY_MISMATCH."""

    route = Route(0, torch.tensor([[10.0, 10.0], [1.0, 0.0]]))
    result = ingest(
        _graph(),
        DrawingScene(
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            (route,),
        ),
        StyleContract(),
        ObservationProfile(),
    )
    assert isinstance(result, InvalidScene)
    assert result.code is IngestionErrorCode.TOPOLOGY_MISMATCH


def test_producer_extent_field_is_rejected() -> None:
    """Untrusted records cannot choose primitive extents."""

    result = ingest_record(
        _graph(),
        {"positions": [[0, 0], [1, 0], [2, 0]], "half_extents": [[1, 1]] * 3},
        StyleContract(),
        ObservationProfile(),
    )
    assert isinstance(result, InvalidScene)
    assert result.code is IngestionErrorCode.PRODUCER_EXTENTS_FORBIDDEN


def test_topology_mismatch_is_invalid() -> None:
    """Position count must match the canonical graph exactly."""

    result = ingest(
        _graph(),
        DrawingScene(torch.tensor([[0.0, 0.0], [1.0, 0.0]])),
        StyleContract(),
        ObservationProfile(),
    )
    assert isinstance(result, InvalidScene)
    assert result.code is IngestionErrorCode.TOPOLOGY_MISMATCH


def test_coordinate_unit_reexpression_preserves_profile_identity() -> None:
    """Scaling positions and dimensional style together preserves profile identity."""

    graph = _graph()
    base = ingest(
        graph,
        DrawingScene(torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])),
        StyleContract(),
        ObservationProfile(),
    )
    scaled = ingest(
        graph,
        DrawingScene(torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])),
        StyleContract(coordinate_scale=10.0),
        ObservationProfile(),
    )
    assert isinstance(base, ValidScene)
    assert isinstance(scaled, ValidScene)
    assert base.scene.profile_hash == scaled.scene.profile_hash
    assert scaled.scene.intrinsic_unit == 10.0 * base.scene.intrinsic_unit


def test_nonunit_declared_axis_is_invalid() -> None:
    """Declared axes are graph-owned unit vectors, never drawing-derived hints."""

    graph = GraphSemantics(("a", "b"), ((0, 1),), directed=True, flow_axis=(0.0, 2.0))
    result = ingest(
        graph,
        DrawingScene(torch.tensor([[0.0, 0.0], [0.0, 1.0]])),
        StyleContract(),
        ObservationProfile(),
    )
    assert isinstance(result, InvalidScene)
    assert result.code is IngestionErrorCode.MALFORMED_METADATA


def test_nonautomorphic_symmetry_generator_is_invalid() -> None:
    """A permutation that does not preserve topology cannot activate U06."""

    graph = GraphSemantics(
        ("a", "b", "c"),
        ((0, 1),),
        symmetry_generators=((1, 2, 0),),
    )
    result = ingest(
        graph,
        DrawingScene(torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])),
        StyleContract(),
        ObservationProfile(),
    )
    assert isinstance(result, InvalidScene)
    assert result.code is IngestionErrorCode.MALFORMED_METADATA


def test_nonmonotone_thickness_map_is_invalid() -> None:
    """Thickness encoding knots must be positive and monotone at ingestion."""

    graph = GraphSemantics(
        ("a", "b", "c"),
        ((0, 1), (1, 2)),
        edge_weights=(1.0, 2.0),
        weight_semantics="connection_strength",
        weight_visual_channel="stroke_thickness",
        weight_encoding_knots=((1.0, 2.0), (2.0, 1.0)),
    )
    result = ingest(
        graph,
        DrawingScene(torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])),
        StyleContract(edge_stroke_widths=(2.0, 1.0)),
        ObservationProfile(),
    )
    assert isinstance(result, InvalidScene)
    assert result.code is IngestionErrorCode.MALFORMED_METADATA


def test_declared_thickness_with_missing_derived_width_is_invalid() -> None:
    """A declared thickness channel cannot omit any corpus-derived edge stroke."""

    graph = GraphSemantics(
        ("a", "b"),
        ((0, 1),),
        edge_weights=(1.0,),
        weight_semantics="connection_strength",
        weight_visual_channel="stroke_thickness",
        weight_encoding_knots=((1.0, 1.0),),
    )
    result = ingest(
        graph,
        DrawingScene(torch.tensor([[0.0, 0.0], [1.0, 0.0]])),
        StyleContract(),
        ObservationProfile(),
    )
    assert isinstance(result, InvalidScene)
    assert result.code is IngestionErrorCode.MISSING_REQUIRED_PRIMITIVE


def test_multiedge_missing_distinct_route_is_invalid() -> None:
    """Every edge id in a parallel class requires its own visible route."""

    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    graph = GraphSemantics(("a", "b"), ((0, 1), (0, 1)))
    result = ingest(
        graph,
        DrawingScene(positions, (Route(0, positions.clone()),)),
        StyleContract(),
        ObservationProfile(),
    )
    assert isinstance(result, InvalidScene)
    assert result.code is IngestionErrorCode.MISSING_REQUIRED_PRIMITIVE


def test_malformed_temporal_magnitude_is_invalid() -> None:
    """An unchanged temporal node cannot declare positive expected displacement."""

    graph = GraphSemantics(
        ("a", "b", "c"),
        ((0, 1), (0, 2)),
        temporal_ids=("a", "b", "c"),
    )
    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    first = ingest(graph, DrawingScene(positions), StyleContract(), ObservationProfile())
    second = ingest(graph, DrawingScene(positions), StyleContract(), ObservationProfile())
    assert isinstance(first, ValidScene)
    assert isinstance(second, ValidScene)
    transition = TemporalTransition(
        {"a": "unchanged", "b": "unchanged", "c": "unchanged"},
        {"a": 1.0, "b": 0.0, "c": 0.0},
    )
    result = ingest_temporal((first.scene, second.scene), (transition,))
    assert isinstance(result, InvalidScene)
    assert result.code is IngestionErrorCode.MALFORMED_METADATA


def test_valid_temporal_scene_preserves_typed_transition() -> None:
    """Consistent temporal identities and magnitudes produce a validated sequence."""

    graph = GraphSemantics(
        ("a", "b", "c"),
        ((0, 1), (0, 2)),
        temporal_ids=("a", "b", "c"),
    )
    positions = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    first = ingest(graph, DrawingScene(positions), StyleContract(), ObservationProfile())
    second = ingest(graph, DrawingScene(positions), StyleContract(), ObservationProfile())
    assert isinstance(first, ValidScene)
    assert isinstance(second, ValidScene)
    transition = TemporalTransition(
        {"a": "unchanged", "b": "unchanged", "c": "unchanged"},
        {"a": 0.0, "b": 0.0, "c": 0.0},
    )
    result = ingest_temporal((first.scene, second.scene), (transition,))
    assert isinstance(result, ValidTemporalScene)
