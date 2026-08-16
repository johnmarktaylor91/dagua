"""Falsifiable rank-fidelity certification on a discriminating population.

The pilot bank cannot falsify the 6.5 tau floor: the traced surrogate is
the frozen closed forms executed on tensors, so its forward value differs
from exact by 1-3 ULP (measured max 3.3e-16) while the smallest
inter-scene loss gap in any bank cell is 2.8e-4 -- twelve-plus orders of
magnitude wider than any deviation that could flip a pair. Bank tau is
therefore 1.0 by construction and is published as VACUOUS (DISCREPANCIES
entry 39), not as evidence.

This module supplies the evidence the bank cannot: a graded loss-gap
LADDER population whose bottom rungs are separated by single ULPs of
``l_total`` -- gaps at or below the sum of the per-scene soft-vs-exact
deviations, where a rank disagreement is structurally POSSIBLE. Both
gates here can fail:

- the ladder gate fails if traced-vs-exact deviations misorder scenes
  whose gaps are within reach of the deviation scale (a discriminating
  precondition assertion proves the population contains such pairs);
- the drift-control gate fails if the certification machinery ever stops
  detecting a drifted surrogate (the P5 failure mode the 6.3 tau floor
  is re-scoped to catch: a distilled or partially detached surrogate).
"""

from __future__ import annotations

import pytest
import torch

from dagua.eval.ruler_v4.certification import certify_rank_fidelity
from dagua.eval.ruler_v4.composition import compose
from dagua.eval.ruler_v4.score import _evaluate_static_facets
from dagua.eval.ruler_v4.surrogate.traced import score_scene_soft
from tests.eval.ruler_v4.scene_bank import _ingest_variant, build_cell
from tests.eval.ruler_v4.test_certification_population import _profiles, _table

TAU_THRESHOLD = 0.85
# Target exact-loss gaps from clearly-separated down to single ULPs of a
# loss near 0.4. The bottom rungs quantize to 1-4 ULP position-rounded
# gaps; rungs whose perturbation rounds away entirely produce exact ties
# and are excluded from comparable pairs by certify_rank_fidelity.
_TARGET_GAPS = (
    1e-3,
    1e-5,
    1e-7,
    1e-9,
    1e-11,
    1e-12,
    1e-13,
    3e-14,
    1e-14,
    3e-15,
    1e-15,
    7e-16,
    5e-16,
    3e-16,
    2e-16,
)


@pytest.fixture(scope="module")
def ladder_losses():
    """Score the gap-ladder population on both paths once per module.

    Returns
    -------
    tuple[list[float], list[float], list[float]]
        Exact losses, soft losses, and per-scene absolute deviations, in
        ladder order (base scene first).
    """

    table = _table()
    profiles = _profiles()
    base = build_cell("chain", "small").scenes[3]  # noise_moderate: generic geometry
    traced = score_scene_soft(base, table, profiles)
    (gradient,) = torch.autograd.grad(traced.soft.l_total, traced.positions)
    norm = float(torch.linalg.vector_norm(gradient))
    assert norm > 0.0, "ladder base scene has a dead l_total gradient"
    direction = gradient / norm

    scenes = [base]
    for gap in _TARGET_GAPS:
        positions = base.positions.detach() + (gap / norm) * direction
        scenes.append(_ingest_variant(base.graph, positions))

    exact_losses = []
    soft_losses = []
    deviations = []
    for scene in scenes:
        facets = _evaluate_static_facets(scene, profiles)
        exact = compose(facets, table, profiles.composition).l_total
        soft = float(
            score_scene_soft(scene, table, profiles, exact_facets=facets).soft.l_total.detach()
        )
        exact_losses.append(exact)
        soft_losses.append(soft)
        deviations.append(abs(exact - soft))
    return exact_losses, soft_losses, deviations


@pytest.mark.slow
def test_population_is_discriminating(ladder_losses) -> None:
    """The ladder contains pairs where soft-vs-exact CAN disagree.

    Precondition for the tau gate below to be falsifiable: the surrogate
    is not an identity map (some scene deviates), and at least two pairs
    have an exact gap within the deviation scale (gap <= 4 * max
    deviation), so a deviation of the measured size could flip their
    order. If the traced seam ever becomes bitwise identical to exact,
    this fails loudly: the gate would be vacuous again and must be
    re-published, not silently re-passed.
    """

    exact_losses, _, deviations = ladder_losses
    max_deviation = max(deviations)
    assert max_deviation > 0.0, (
        "soft path is bitwise identical to exact on every ladder scene; "
        "the tau gate is vacuous again -- re-publish before re-asserting"
    )
    flippable = 0
    for i in range(len(exact_losses)):
        for j in range(i + 1, len(exact_losses)):
            gap = abs(exact_losses[i] - exact_losses[j])
            if 0.0 < gap <= 4.0 * max_deviation:
                flippable += 1
    assert flippable >= 2, (
        f"only {flippable} pairs sit within the deviation scale "
        f"(max deviation {max_deviation:.3e}); the population cannot disagree"
    )


@pytest.mark.slow
def test_ladder_rank_fidelity_certifies(ladder_losses) -> None:
    """tau >= 0.85 holds on the population that can actually falsify it.

    The measured traced deviation is a systematic per-graph accumulation
    offset, so it cancels in comparisons and rank agreement holds even at
    1-ULP gaps; a surrogate whose deviation were noise-like at the same
    magnitude could flip the sub-deviation pairs and fail this gate.
    """

    exact_losses, soft_losses, _ = ladder_losses
    fidelity = certify_rank_fidelity(exact_losses, soft_losses, threshold=TAU_THRESHOLD)
    assert fidelity.comparable_pairs >= 20
    assert fidelity.certified, (
        f"ladder tau {fidelity.tau} below {TAU_THRESHOLD} with "
        f"{fidelity.discordant_pairs} discordant of {fidelity.comparable_pairs} pairs"
    )


@pytest.mark.slow
def test_drift_control_fails_certification(ladder_losses) -> None:
    """A drifted surrogate is refused: the gate mechanism CAN emit failure.

    Quantizing the soft losses to 3 decimals stands in for the P5 drift
    class the 6.3 tau floor is re-scoped to catch (a distilled or
    partially detached surrogate whose forward value no longer tracks
    exact). The certification must come back uncertified on the same
    population that certifies the delivered surrogate.
    """

    exact_losses, soft_losses, _ = ladder_losses
    drifted = [round(value, 3) for value in soft_losses]
    fidelity = certify_rank_fidelity(exact_losses, drifted, threshold=TAU_THRESHOLD)
    assert not fidelity.certified, (
        "a 3-decimal quantized surrogate still certified: the tau gate "
        f"cannot detect drift (tau {fidelity.tau}, "
        f"{fidelity.comparable_pairs} comparable pairs)"
    )
