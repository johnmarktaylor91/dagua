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


def test_dof_account_models_the_a18_allocation_buckets() -> None:
    """Identities spend named A18 buckets; unassigned identities fail closed."""

    from dagua.eval.ruler_v4.weights import DOF_ALLOCATION

    assert sum(DOF_ALLOCATION.values()) == 20
    table = WeightTable(
        entries=(
            SubtermWeight("A.1", "A", "test", 0.6, fitted_parameter="w_A"),
            SubtermWeight("B.1", "B", "test", 0.4, fitted_parameter="w_B"),
        ),
        d_power=20,
        other_fitted_parameters=("tail_blend",),
        fitted_parameter_buckets={
            "w_A": "universal",
            "w_B": "universal",
            "tail_blend": "aggregation",
        },
    )
    account = table.dof_account
    assert account.bucket_usage == {"universal": 2, "aggregation": 1}
    assert account.within_buckets
    assert account.unassigned_identities == ()

    unassigned = WeightTable(
        entries=(SubtermWeight("A.1", "A", "test", 1.0, fitted_parameter="w_A"),),
        d_power=20,
    )
    assert unassigned.dof_account.unassigned_identities == ("w_A",)
    assert not unassigned.dof_account.within_buckets


def test_unspent_reserve_and_unknown_buckets_are_rejected() -> None:
    """The A18 UNSPENT reserve is headroom, never an assignable bucket."""

    for bucket in ("unspent", "made_up"):
        with pytest.raises(ValueError, match="unknown or reserved A18 bucket"):
            WeightTable(
                entries=(SubtermWeight("A.1", "A", "test", 1.0, fitted_parameter="w_A"),),
                d_power=20,
                fitted_parameter_buckets={"w_A": bucket},
            )


def test_bucket_overspend_fails_the_contract_gate() -> None:
    """A table spending more identities than a bucket's allocation is refused."""

    from dagua.eval.ruler_v4.weights import DOF_ALLOCATION

    width = DOF_ALLOCATION["semantic"] + 1
    table = WeightTable(
        entries=tuple(
            SubtermWeight(f"A.{index}", "A", "test", 1.0, fitted_parameter=f"w_{index}")
            for index in range(width)
        ),
        d_power=20,
        fitted_parameter_buckets={f"w_{index}": "semantic" for index in range(width)},
    )
    assert not table.dof_account.within_buckets
    assert table.dof_account.bucket_usage == {"semantic": width}


def test_provenance_class_is_validated_and_bundle_consistent() -> None:
    """Fitted provenance requires an identity and bundles cannot mix classes."""

    with pytest.raises(ValueError, match="fitted provenance requires a fitted identity"):
        WeightTable(
            entries=(SubtermWeight("A.1", "A", "test", 1.0, provenance_class="fitted"),),
            d_power=20,
        )
    with pytest.raises(ValueError, match="a fitted identity requires fitted provenance"):
        WeightTable(
            entries=(
                SubtermWeight(
                    "A.1",
                    "A",
                    "test",
                    1.0,
                    fitted_parameter="w_A",
                    provenance_class="preregistered_prior",
                ),
            ),
            d_power=20,
        )
    with pytest.raises(ValueError, match="bundles mix provenance classes"):
        WeightTable(
            entries=(
                SubtermWeight(
                    "A.1", "A", "test", 0.6, fitted_parameter="w_A", provenance_class="fitted"
                ),
                SubtermWeight(
                    "A.2",
                    "A",
                    "test",
                    0.4,
                    fitted_parameter="w_A",
                    provenance_class="controlled_stimulus",
                ),
            ),
            d_power=20,
        )
