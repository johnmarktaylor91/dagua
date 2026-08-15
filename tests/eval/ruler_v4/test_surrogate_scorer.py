"""Tests for the compiled differentiable V4 surrogate scorer."""

from __future__ import annotations

from typing import Dict, Tuple

import pytest
import torch

from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile, compose
from dagua.eval.ruler_v4.scene import FacetResult, value_result
from dagua.eval.ruler_v4.surrogate import score_v4_soft
from dagua.eval.ruler_v4.weight_table import SubtermWeight, WeightTable


def _three_term_inputs(
    values: Tuple[float, float, float],
) -> Tuple[Dict[str, FacetResult], WeightTable]:
    """Build a three-term composition probe.

    Parameters
    ----------
    values : tuple[float, float, float]
        Exact defects for U01, U03.r_1, and U07.base.

    Returns
    -------
    tuple[dict[str, FacetResult], WeightTable]
        Facet results and explicit weights.
    """

    facets = {
        "U01": value_result(values[0], {"U01.headline": values[0]}),
        "U03": value_result(values[1], {"U03.r_1": values[1]}),
        "U07": value_result(values[2], {"U7.base": values[2]}),
    }
    table = WeightTable(
        entries=(
            SubtermWeight("U01.headline", "U01", "G1", 1.0),
            SubtermWeight("U03.r_1", "U03", "G1", 2.0),
            SubtermWeight("U7.base", "U07", "G2", 1.0),
        ),
        d_power=20,
    )
    return facets, table


@pytest.mark.parametrize("power", [1.0, 2.0, 4.0])
def test_soft_forward_matches_exact_p_mean(power: float) -> None:
    """Identity-bound compiled rows reproduce exact p-mean loss."""

    facets, table = _three_term_inputs((0.2, 0.5, 0.8))
    profile = CompositionProfile(CompositionFamily.P_MEAN, power=power)

    soft = score_v4_soft(facets, table, profile)
    exact = compose(facets, table, profile)

    assert float(soft.l_mean) == pytest.approx(exact.l_mean, abs=1e-15)
    assert float(soft.l_total) == pytest.approx(exact.l_total, abs=1e-15)
    expected_ids = {"U01.headline", "U03.r_1", "U7.base"}
    assert all(term.trace.subterm_id in expected_ids for term in soft.terms)


def test_soft_gradient_matches_finite_difference() -> None:
    """Autograd agrees with a central finite difference on defect probes."""

    facets, table = _three_term_inputs((0.2, 0.5, 0.8))
    profile = CompositionProfile(CompositionFamily.P_MEAN, power=2.0)
    probe = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    soft = score_v4_soft(
        facets,
        table,
        profile,
        term_tensors={"U03.r_1": probe},
    )
    soft.l_total.backward()
    assert probe.grad is not None

    epsilon = 1.0e-6
    upper_facets, _ = _three_term_inputs((0.2, 0.5 + epsilon, 0.8))
    lower_facets, _ = _three_term_inputs((0.2, 0.5 - epsilon, 0.8))
    finite_difference = (
        compose(upper_facets, table, profile).l_total
        - compose(lower_facets, table, profile).l_total
    ) / (2.0 * epsilon)

    assert float(probe.grad) == pytest.approx(finite_difference, rel=1e-8, abs=1e-10)


def test_negative_soft_gradient_improves_exact_facet_and_composite() -> None:
    """A small negative-gradient move lowers the exact bound defect and score."""

    facets, table = _three_term_inputs((0.2, 0.5, 0.8))
    profile = CompositionProfile(CompositionFamily.P_MEAN, power=2.0)
    probe = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    soft = score_v4_soft(
        facets,
        table,
        profile,
        term_tensors={"U03.r_1": probe},
    )
    soft.l_total.backward()
    assert probe.grad is not None
    moved = float(probe.detach() - 0.05 * probe.grad)

    improved_facets, _ = _three_term_inputs((0.2, moved, 0.8))
    assert moved < 0.5
    assert compose(improved_facets, table, profile).l_total < soft.exact_composition.l_total


def test_soft_bottleneck_forward_and_gradient_are_finite() -> None:
    """The optional exact composition family remains analytic above onset."""

    facets, table = _three_term_inputs((0.2, 0.5, 0.8))
    profile = CompositionProfile(
        CompositionFamily.MEAN_SOFT_BOTTLENECK,
        bottleneck_mix=0.4,
        bottleneck_temperature=0.05,
        group_allowances={"G1": 0.1, "G2": 0.1},
    )
    probe = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    soft = score_v4_soft(
        facets,
        table,
        profile,
        term_tensors={"U7.base": probe},
    )
    soft.l_total.backward()

    assert float(soft.l_total.detach()) == pytest.approx(
        soft.exact_composition.l_total,
        abs=1e-15,
    )
    assert probe.grad is not None and bool(torch.isfinite(probe.grad).item())
