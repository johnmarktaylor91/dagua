"""Banked stratified rank-fidelity certification on the pilot bank.

Replaces the single-fixture smoke's population with the 6.3-shaped
stratification contract at test scale: structural classes x size band,
graded drawing quality inside every cell, per-cell tau gates measured on
the SOFT path (the traced surrogate's l_total, which differs from exact
by construction), and fail-closed comparability. The full pilot numbers
(all classes and bands, plus the 6.2a conformance numbers) are produced
by the population driver recorded in DISCREPANCIES entry 39; this test
keeps the contract permanently red-green.
"""

from __future__ import annotations

import pytest
import torch

from dagua.eval.ruler_v4.certification import certify_rank_fidelity
from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile, compose
from dagua.eval.ruler_v4.contracts import CONTRACTS
from dagua.eval.ruler_v4.headline import HeadlineProfile
from dagua.eval.ruler_v4.score import ScoringProfiles, _evaluate_static_facets
from dagua.eval.ruler_v4.surrogate.traced import score_scene_soft
from dagua.eval.ruler_v4.weight_table import (
    GATE_DIAGNOSTIC_FACETS,
    REQUIRED_PRIOR_FLOOR_FACETS,
    ParameterProvenance,
    SubtermWeight,
    WeightTable,
)
from tests.eval.ruler_v4.scene_bank import bank_digest, build_cell

TAU_THRESHOLD = 0.85
_CELLS = (("chain", "small"), ("clustered", "small"))


def _table() -> WeightTable:
    """Build the complete fixed test table."""

    entries = tuple(
        SubtermWeight(
            subterm_id=subterm_id,
            facet_id=facet_id,
            group=facet_id[:3],
            weight=0.0 if facet_id in GATE_DIAGNOSTIC_FACETS else 1.0,
            prior_driven=facet_id in REQUIRED_PRIOR_FLOOR_FACETS,
            diagnostic=facet_id in GATE_DIAGNOSTIC_FACETS,
            provenance_class=(
                None
                if facet_id in GATE_DIAGNOSTIC_FACETS
                else "preregistered_prior"
                if facet_id in REQUIRED_PRIOR_FLOOR_FACETS
                else "contract_frozen"
            ),
        )
        for facet_id, contract in CONTRACTS.items()
        for subterm_id in contract.scored_subterms
    )
    return WeightTable(
        entries=entries,
        d_power=20,
        prior_floors={facet_id: 1.0 for facet_id in REQUIRED_PRIOR_FLOOR_FACETS},
    )


def _profiles() -> ScoringProfiles:
    """Build explicit non-fitted profiles."""

    return ScoringProfiles(
        composition=CompositionProfile(CompositionFamily.P_MEAN, power=2.0),
        headline=HeadlineProfile(index_span=100.0, loss_scale=1.0, version="bank"),
        measurement_version="bank-measurement",
        policy_version="bank-policy",
        alpha_grid_index=1,
        parameter_provenance={
            "composition.power": ParameterProvenance("preregistered_prior"),
            "headline.index_span": ParameterProvenance("contract_frozen"),
            "headline.loss_scale": ParameterProvenance("preregistered_prior"),
            "alpha_grid_index": ParameterProvenance("preregistered_prior"),
        },
    )


def test_bank_is_deterministic() -> None:
    """The pilot bank digest is stable across constructions."""

    first = [build_cell(*cell) for cell in _CELLS]
    second = [build_cell(*cell) for cell in _CELLS]
    assert bank_digest(first) == bank_digest(second)


@pytest.mark.slow
@pytest.mark.parametrize("class_name,scale_name", _CELLS)
def test_soft_path_rank_fidelity_per_cell(class_name: str, scale_name: str) -> None:
    """Per-cell tau on the SOFT path clears the 0.85 contract with real pairs."""

    table = _table()
    profiles = _profiles()
    cell = build_cell(class_name, scale_name)
    exact_losses = []
    soft_losses = []
    for scene in cell.scenes:
        facets = _evaluate_static_facets(scene, profiles)
        exact_losses.append(compose(facets, table, profiles.composition).l_total)
        traced = score_scene_soft(scene, table, profiles, exact_facets=facets)
        # The certified quantity is the surrogate's own forward value: the
        # tensor recomposition, not a re-read of the exact float.
        soft_losses.append(float(traced.soft.l_total.detach()))
        assert traced.bound_subterms, "soft path bound no traced terms; surrogate is inert"
        assert traced.soft.l_total.requires_grad
    fidelity = certify_rank_fidelity(exact_losses, soft_losses, threshold=TAU_THRESHOLD)
    assert fidelity.comparable_pairs > 0
    assert fidelity.certified, (
        f"[{class_name}/{scale_name}] tau {fidelity.tau} below {TAU_THRESHOLD} "
        f"({fidelity.comparable_pairs} comparable pairs)"
    )


@pytest.mark.slow
def test_soft_path_l_total_gradient_live_on_bank() -> None:
    """The traced l_total carries a nonzero position gradient on bank scenes."""

    table = _table()
    profiles = _profiles()
    cell = build_cell("layered_dag", "small")
    scene = cell.scenes[3]  # moderate noise: defects present, geometry generic
    traced = score_scene_soft(scene, table, profiles)
    (gradient,) = torch.autograd.grad(traced.soft.l_total, traced.positions)
    assert bool(torch.isfinite(gradient).all())
    assert float(torch.linalg.vector_norm(gradient)) > 0.0
