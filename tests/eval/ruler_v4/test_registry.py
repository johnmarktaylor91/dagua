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
        assert set(result.subterms) == set(CONTRACTS[facet_id].scored_subterms)


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
    else:
        assert result.value is not None or result.state is ResultState.INVALID


@pytest.mark.parametrize("facet_id", CONTRACTS)
def test_contract_invalid_ingestion_path_is_typed(facet_id: str) -> None:
    """Reject malformed geometry before dispatch for every contract entry point."""

    del facet_id
    graph = GraphSemantics(("a", "b"), ((0, 1),))
    result = ingest_record(
        graph,
        {"positions": [[0.0, 0.0], [float("nan"), 1.0]]},
        StyleContract(),
        ObservationProfile(),
    )
    assert isinstance(result, InvalidScene)
    assert result.code is IngestionErrorCode.NONFINITE_GEOMETRY
