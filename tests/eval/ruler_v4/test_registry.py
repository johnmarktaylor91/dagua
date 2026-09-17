"""Contract inventory and independent facet dispatch tests."""

from __future__ import annotations

import math

import pytest

from dagua.eval.ruler_v4.contracts import CONTRACTS, SCORED_SUBTERM_COUNT
from dagua.eval.ruler_v4.ingestion import ingest_record
from dagua.eval.ruler_v4.registry import FACET_FUNCTIONS, evaluate_facet, validate_registry
from dagua.eval.ruler_v4.scene import (
    GraphSemantics,
    IngestionErrorCode,
    InvalidScene,
    ObservationProfile,
    ResultState,
    Scene,
    StyleContract,
)


def test_frozen_contract_inventory() -> None:
    """Require all 45 contracts and exactly 91 score-visible sub-terms."""

    validate_registry()
    assert len(CONTRACTS) == 45
    assert SCORED_SUBTERM_COUNT == 91
    assert set(FACET_FUNCTIONS) == set(CONTRACTS)


@pytest.mark.parametrize("facet_id", CONTRACTS)
def test_contract_smoke_case(facet_id: str, semantic_scene: Scene) -> None:
    """Execute every exact-id facet on one semantics-rich valid scene."""

    result = evaluate_facet(facet_id, semantic_scene)
    assert result.state in {ResultState.VALUE, ResultState.NA, ResultState.INVALID}
    if result.value is not None:
        assert math.isfinite(result.value)
        assert 0.0 <= result.value <= 1.0
        assert set(result.subterms).issubset(CONTRACTS[facet_id].scored_subterms)


def test_u07_dispatch_forwards_fitted_parameters(semantic_scene: Scene) -> None:
    """Expose both fitted U07 scalars through the public facet registry."""

    result = evaluate_facet("U07", semantic_scene, gamma=2.0, lambda_T=0.25)
    assert result.raw["gamma"] == 2.0
    assert result.raw["lambda_T"] == 0.25


@pytest.mark.parametrize("facet_id", CONTRACTS)
def test_contract_worked_identity(facet_id: str) -> None:
    """Pin each contract title, exact id, and frozen hash used by worked goldens."""

    function = FACET_FUNCTIONS[facet_id]
    metadata = CONTRACTS[facet_id]
    assert function.__name__ == facet_id
    assert metadata.title in (function.__doc__ or "")
    assert metadata.sha256 in (function.__doc__ or "")


@pytest.mark.parametrize("facet_id", CONTRACTS)
def test_contract_na_case_is_typed(facet_id: str, semantic_scene: Scene) -> None:
    """Every contract exposes a typed state and never returns a silent NaN."""

    result = evaluate_facet(facet_id, semantic_scene)
    if result.state is ResultState.NA:
        assert result.value is None
        assert result.reason
    elif result.state is ResultState.VALUE:
        assert result.subterms
    else:
        assert result.state is ResultState.INVALID
        assert result.value is None
        assert result.reason


def test_invalid_ingestion_precedes_facet_dispatch() -> None:
    """Reject malformed geometry once, before any facet can be dispatched."""

    graph = GraphSemantics(("a", "b"), ((0, 1),))
    result = ingest_record(
        graph,
        {"positions": [[0.0, 0.0], [float("nan"), 1.0]]},
        StyleContract(),
        ObservationProfile(),
    )
    assert isinstance(result, InvalidScene)
    assert result.code is IngestionErrorCode.NONFINITE_GEOMETRY
