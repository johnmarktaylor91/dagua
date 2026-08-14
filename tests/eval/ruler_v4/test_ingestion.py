"""Three-way ingestion contract tests."""

from __future__ import annotations

import torch

from dagua.eval.ruler_v4.ingestion import ingest, ingest_record
from dagua.eval.ruler_v4.scene import (
    DrawingScene,
    GraphSemantics,
    IngestionErrorCode,
    InvalidScene,
    ObservationProfile,
    StyleContract,
    ValidAbsence,
    ValidScene,
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


def test_optional_absence_is_typed_na() -> None:
    """A profile-wide optional missing channel produces ValidAbsence."""

    profile = ObservationProfile(optional_channels=frozenset({"node_labels"}))
    result = ingest(
        _graph(),
        DrawingScene(torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])),
        StyleContract(),
        profile,
    )
    assert isinstance(result, ValidAbsence)
    assert result.reason == "no_declared_node_labels"


def test_required_absence_is_invalid() -> None:
    """A required missing route is invalid rather than favorable NA."""

    graph = GraphSemantics(
        ("a", "b"), ((0, 1),), required_primitives=frozenset({"nodes", "routes"})
    )
    result = ingest(
        graph,
        DrawingScene(torch.tensor([[0.0, 0.0], [1.0, 0.0]])),
        StyleContract(),
        ObservationProfile(),
    )
    assert isinstance(result, InvalidScene)
    assert result.code is IngestionErrorCode.MISSING_REQUIRED_PRIMITIVE


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
