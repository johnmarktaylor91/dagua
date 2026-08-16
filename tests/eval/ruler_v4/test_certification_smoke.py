"""Fixed-seed determinism smoke of the identity-surrogate scoring pipeline.

This is a FLOAT-DETERMINISM check, not certification evidence: with no
``term_tensors`` bound, ``score_v4_soft`` recomposes the exact facet
values, so ranking its output against the exact scorer compares f(x)
with f(x) and tau is 1.0 for ANY batch by construction. The smoke pins
only that two identically-seeded end-to-end runs produce identical
bytes. Rank-fidelity evidence for the traced surrogate lives in
test_certification_population.py (bank machinery) and
test_certification_discriminating.py (the falsifiable population).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Mapping, Tuple

import pytest
import torch

from dagua.eval.ruler_v4.certification import RankFidelityResult, certify_scene_batch
from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile
from dagua.eval.ruler_v4.contracts import CONTRACTS
from dagua.eval.ruler_v4.headline import HeadlineProfile
from dagua.eval.ruler_v4.ingestion import ingest
from dagua.eval.ruler_v4.scene import DrawingScene, FacetResult, Route, Scene, ValidScene
from dagua.eval.ruler_v4.score import ScoreResult, ScoringProfiles, score
from dagua.eval.ruler_v4.surrogate import score_v4_soft
from dagua.eval.ruler_v4.weight_table import (
    GATE_DIAGNOSTIC_FACETS,
    REQUIRED_PRIOR_FLOOR_FACETS,
    ParameterProvenance,
    SubtermWeight,
    WeightTable,
)


def _complete_table() -> WeightTable:
    """Build the complete fixed test table for the true scorer.

    Returns
    -------
    WeightTable
        Valid 91-row table with explicit provenance.
    """

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
    """Build explicit non-fitted profiles for the smoke.

    Returns
    -------
    ScoringProfiles
        Deterministic composition and headline profiles.
    """

    return ScoringProfiles(
        composition=CompositionProfile(CompositionFamily.P_MEAN, power=2.0),
        headline=HeadlineProfile(index_span=100.0, loss_scale=1.0, version="cert-smoke"),
        measurement_version="cert-smoke-measurement",
        policy_version="cert-smoke-policy",
        alpha_grid_index=1,
        parameter_provenance={
            "composition.power": ParameterProvenance("preregistered_prior"),
            "headline.index_span": ParameterProvenance("contract_frozen"),
            "headline.loss_scale": ParameterProvenance("preregistered_prior"),
            "alpha_grid_index": ParameterProvenance("preregistered_prior"),
        },
    )


def _generated_scenes(base: Scene, seed: int) -> Tuple[Scene, ...]:
    """Generate a tiny deterministic batch from one validated scene.

    Parameters
    ----------
    base : Scene
        Broad semantic fixture.
    seed : int
        Fixed torch generator seed.

    Returns
    -------
    tuple[Scene, ...]
        Four re-ingested position perturbations.
    """

    graph = replace(
        base.graph,
        tree_parents=(None, 0, 0, 1, 2, 3, 4, 5),
        tree_depths=(0, 1, 1, 2, 2, 3, 3, 4),
        tree_layout="layered",
        flow_axis=(1.0, 0.0),
        ordered_children={0: (1, 2)},
    )
    generator = torch.Generator().manual_seed(seed)
    scenes: List[Scene] = []
    for index, scale in enumerate((0.85, 1.0, 1.15, 1.30)):
        noise = torch.randn(
            base.positions.shape,
            generator=generator,
            dtype=torch.float64,
        ) * (0.01 * index)
        positions = base.positions * scale + noise
        routes = tuple(
            Route(edge_index, torch.stack((positions[source], positions[target])))
            for edge_index, (source, target) in enumerate(graph.edges)
        )
        result = ingest(
            graph,
            DrawingScene(positions, routes, base.z_order),
            base.style,
            base.profile,
        )
        assert isinstance(result, ValidScene)
        scenes.append(result.scene)
    return tuple(scenes)


def _certify_batch(base: Scene, seed: int) -> RankFidelityResult:
    """Score and certify one generated batch.

    Parameters
    ----------
    base : Scene
        Source scene.
    seed : int
        Batch seed.

    Returns
    -------
    RankFidelityResult
        True-vs-compiled ranking certificate.
    """

    scenes = _generated_scenes(base, seed)
    table = _complete_table()
    profiles = _profiles()
    exact: Dict[int, ScoreResult] = {id(scene): score(scene, table, profiles) for scene in scenes}

    def true_scorer(scene: Scene) -> float:
        """Return cached exact pre-map loss for one generated scene."""

        return exact[id(scene)].type_m.composition.l_total

    def surrogate_scorer(scene: Scene) -> float:
        """Return compiled identity-surrogate loss for one generated scene."""

        facet_results: Mapping[str, FacetResult] = {
            facet_id: breakdown.result
            for facet_id, breakdown in exact[id(scene)].type_m.facets.items()
        }
        return float(
            score_v4_soft(
                facet_results,
                table,
                profiles.composition,
            ).l_total
        )

    return certify_scene_batch(
        scenes,
        true_scorer,
        surrogate_scorer,
        threshold=0.85,
    )


@pytest.mark.smoke
def test_identity_surrogate_scoring_is_deterministic(semantic_scene: Scene) -> None:
    """Two identically-seeded end-to-end scoring runs are byte-identical.

    No tau or certification assertion belongs here: the identity
    surrogate's tau is definitionally 1.0 and certifies nothing
    (P3REVIEW2 OPUS5 MAJOR-4).
    """

    first = _certify_batch(semantic_scene, seed=20260815)
    second = _certify_batch(semantic_scene, seed=20260815)

    assert first == second
