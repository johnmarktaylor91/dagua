"""Tests for V4 fitted-weight, capacity, and disclosure machinery."""

from __future__ import annotations

import pytest

from dagua.eval.ruler_v4.contracts import CONTRACTS
from dagua.eval.ruler_v4.weights import (
    GATE_DIAGNOSTIC_FACETS,
    REQUIRED_PRIOR_FLOOR_FACETS,
    SubtermWeight,
    WeightTable,
)


def _complete_table() -> WeightTable:
    """Build an explicit contract-complete test table.

    Returns
    -------
    WeightTable
        A valid 91-row table with fixed test masses.
    """

    entries = tuple(
        SubtermWeight(
            subterm_id=subterm_id,
            facet_id=facet_id,
            group="test",
            weight=0.0 if facet_id in GATE_DIAGNOSTIC_FACETS else 1.0,
            prior_driven=facet_id in REQUIRED_PRIOR_FLOOR_FACETS,
            diagnostic=facet_id in GATE_DIAGNOSTIC_FACETS,
        )
        for facet_id, contract in CONTRACTS.items()
        for subterm_id in contract.scored_subterms
    )
    return WeightTable(
        entries=entries,
        d_power=20,
        prior_floors={facet_id: 1.0 for facet_id in REQUIRED_PRIOR_FLOOR_FACETS},
    )


def test_contract_complete_table_carries_diagnostics_and_floors() -> None:
    """A complete table validates all frozen DIAG and prior-floor obligations."""

    table = _complete_table()

    table.validate_for_contracts()
    assert len(table.entries) == 91
    assert {entry.facet_id for entry in table.entries if entry.diagnostic} == (
        GATE_DIAGNOSTIC_FACETS
    )
    assert all(entry.weight == 0.0 for entry in table.entries if entry.diagnostic)
    assert set(table.prior_floors) == REQUIRED_PRIOR_FLOOR_FACETS


def test_dof_account_counts_shared_fitted_scalar_once() -> None:
    """A fixed-ratio bundle consumes one independently adjustable dof."""

    table = WeightTable(
        entries=(
            SubtermWeight("A.1", "A", "test", 0.6, fitted_parameter="w_A"),
            SubtermWeight("A.2", "A", "test", 0.4, fitted_parameter="w_A"),
            SubtermWeight("B.1", "B", "test", 1.0, fitted_parameter="w_B"),
        ),
        d_power=2,
        other_fitted_parameters=("aggregation_mix",),
    )

    assert table.dof_account.allowed == 2
    assert table.dof_account.used == 3
    assert table.dof_account.remaining == 0
    assert not table.dof_account.within_cap


def test_prior_mass_disclosure_uses_post_na_mass_and_strict_gate() -> None:
    """PM-1 renormalizes applicable mass and fires only above 15 percent."""

    table = WeightTable(
        entries=(
            SubtermWeight("A.1", "A", "test", 15.0, prior_driven=True),
            SubtermWeight("B.1", "B", "test", 85.0),
            SubtermWeight("C.1", "C", "test", 20.0),
        ),
        d_power=0,
    )

    exact = table.prior_mass_disclosure(frozenset({"A.1", "B.1"}))
    after_na = table.prior_mass_disclosure(frozenset({"A.1", "C.1"}))

    assert exact.fraction == 0.15
    assert not exact.partial
    assert after_na.fraction == 15.0 / 35.0
    assert after_na.partial
    assert after_na.affected_facets == ("A",)


def test_weight_table_rejects_nonzero_diagnostic_mass() -> None:
    """A DIAG row cannot leak positive mass into the headline."""

    with pytest.raises(ValueError, match="diagnostic terms must carry weight zero"):
        WeightTable(
            entries=(SubtermWeight("U02.headline", "U02", "test", 1.0, diagnostic=True),),
            d_power=0,
        )
