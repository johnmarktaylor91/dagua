"""Spec 6.5 acceptance battery for the traced differentiable surrogate.

Three gates, all pinned by V4_SPEC_r4 6.5 and the P3 reviews:

1. ``l_total.backward()`` works end to end on the semantic fixture and
   fills a finite nonzero position gradient -- the executed flip of
   P3REVIEW OPUS5 BLOCKER-1 (``element 0 of tensors does not require
   grad``) at f3cffecd.
2. Per-facet gradient sanity: for every battery subterm (the load-bearing
   U01/U07/U11/U17/U21 plus breadth), moving positions along the NEGATIVE
   gradient of the SOFT subterm strictly improves the EXACT subterm after
   re-ingestion at the SMALLEST step, without annihilating it to zero,
   AND moving along the POSITIVE gradient at the same step strictly
   worsens it (the ascent control P3REVIEW2 OPUS5 MAJOR-5 ordered: it
   separates gradient information from displacement-annihilation --
   the old U7.base fixture sat on a knife edge where any displacement
   zeroed the crossing events and the descent direction was irrelevant).
3. A liveness summary: the fixture's active rows split into live / flat /
   constant, with a floor on the live fraction so a regression that kills
   gradient connectivity fails loudly.
4. Bank descent: the same descent+ascent gate on a REAL pilot-bank scene
   (chain/small scrambled) for all five mandated facets. The full n>=5
   scenes-per-mandated-facet evidence, including medium/large scenes and
   U7 random-direction controls, is banked by the offline probe
   (p3fix5_bank_descent.py, GRIND_SUMMARY); this test keeps one bank
   case permanently red-green.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Mapping, Optional, Tuple

import pytest
import torch

from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile
from dagua.eval.ruler_v4.contracts import CONTRACTS
from dagua.eval.ruler_v4.headline import HeadlineProfile
from dagua.eval.ruler_v4.ingestion import ingest
from dagua.eval.ruler_v4.scene import (
    DrawingScene,
    GraphSemantics,
    ObservationProfile,
    Route,
    Scene,
    StyleContract,
    ValidScene,
)
from dagua.eval.ruler_v4.score import ScoringProfiles, _evaluate_static_facets
from dagua.eval.ruler_v4.surrogate.traced import score_scene_soft
from dagua.eval.ruler_v4.weight_table import (
    GATE_DIAGNOSTIC_FACETS,
    REQUIRED_PRIOR_FLOOR_FACETS,
    ParameterProvenance,
    SubtermWeight,
    WeightTable,
)

_SUBTERM_FACET: Mapping[str, str] = {
    subterm_id: facet_id
    for facet_id, contract in CONTRACTS.items()
    for subterm_id in contract.scored_subterms
}

_DEFAULT_SEED = 20260815

# Battery entries: (subterm_id, noise_sigma, homothety_scale, seed) --
# sigma scales the seeded perturbation so each target has a nonzero exact
# defect AND room to improve; scale sprawls the whole construction
# (scale * (layout + noise)), which is what puts U21's anti-sprawl row
# off its zero plateau. The five mandated load-bearing facets (U01, U07,
# U11, U17, U21) are all represented; the rest add breadth across every
# traced module, including the clusters family.
#
# Three rows were re-fixtured after P3REVIEW2 OPUS5 MAJOR-5 found their
# evidence vacuous, via a search over (sigma, seed) requiring: exact
# defect in [0.02, 0.98], gradient norm >= 1e-4, strict non-annihilating
# improvement under -grad AND strict worsening under +grad at the
# smallest step (p3fix5_fixture_search.py):
# - U7.base 0.65 -> (1.0, seed 99): the old fixture annihilated to
#   exactly 0 at every ladder step and did not worsen under ascent (a
#   knife edge, not alignment). On the new one, descent improves
#   0.331884 -> 0.331787, ascent worsens to 0.331981, and BOTH seeded
#   random directions worsen (0.331916 / 0.433316) -- the direction, not
#   the displacement, carries the improvement.
# - U11.v 0.35 -> 1.0: the old fixture started below 1e-6 with gradient
#   norm 4.6e-10 (anchored zero). New: defect 0.200736, norm 0.61.
# - U17.1 0.35 -> 1.2: the old fixture was saturated at 0.999990. New:
#   0.969482, off the ceiling, with two-sided response.
BATTERY: Tuple[Tuple[str, float, float, int], ...] = (
    ("U01.headline", 0.35, 1.0, _DEFAULT_SEED),
    ("U7.base", 1.0, 1.0, 99),
    ("U11.v", 1.0, 1.0, _DEFAULT_SEED),
    ("U17.1", 1.2, 1.0, _DEFAULT_SEED),
    ("U18.le", 0.8, 1.0, _DEFAULT_SEED),
    ("U21.d_sparse_n", 0.35, 4.0, _DEFAULT_SEED),
    ("U03.r_1", 0.35, 1.0, _DEFAULT_SEED),
    ("U09.headline", 0.35, 1.0, _DEFAULT_SEED),
    ("U12.headline", 0.35, 1.0, _DEFAULT_SEED),
    ("U26.i", 0.35, 1.0, _DEFAULT_SEED),
    ("U27.i", 0.35, 1.0, _DEFAULT_SEED),
    ("U28.iii", 0.35, 1.0, _DEFAULT_SEED),
    ("U31.headline", 0.35, 1.0, _DEFAULT_SEED),
    ("U32.L_iso", 0.35, 1.0, _DEFAULT_SEED),
    ("U34.L_mono", 0.35, 1.0, _DEFAULT_SEED),
    ("U35.headline", 0.35, 1.0, _DEFAULT_SEED),
)

# Every battery row improves at the SMALLEST step (measured histogram
# 0.001: 16, others: 0 -- P3REVIEW2 MEASURED-ADDENDUM), so the old
# any-step ladder was unused slack; the gate is the smallest step, both
# directions.
_STEP = 0.001


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
        headline=HeadlineProfile(index_span=100.0, loss_scale=1.0, version="acceptance"),
        measurement_version="acceptance-measurement",
        policy_version="acceptance-policy",
        alpha_grid_index=1,
        parameter_provenance={
            "composition.power": ParameterProvenance("preregistered_prior"),
            "headline.index_span": ParameterProvenance("contract_frozen"),
            "headline.loss_scale": ParameterProvenance("preregistered_prior"),
            "alpha_grid_index": ParameterProvenance("preregistered_prior"),
        },
    )


def _semantic_graph() -> Tuple[GraphSemantics, torch.Tensor]:
    """Build the broad semantic fixture graph and its constructed layout."""

    positions = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, -2.0],
            [2.0, 2.0],
            [4.0, -2.0],
            [4.0, 2.0],
            [6.0, -2.0],
            [6.0, 2.0],
            [8.0, 0.0],
        ],
        dtype=torch.float64,
    )
    edges: Tuple[Tuple[int, int], ...] = (
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    )
    graph = GraphSemantics(
        node_ids=tuple(f"n{index}" for index in range(8)),
        edges=edges,
        directed=True,
        node_labels=tuple(f"n{index}" for index in range(8)),
        edge_labels=tuple(None for _ in edges),
        clusters={"left": (0, 1, 2, 3), "right": (4, 5, 6, 7), "parent": tuple(range(8))},
        cluster_parents={"left": "parent", "right": "parent"},
        ranks=(0, 1, 1, 2, 2, 3, 3, 4),
        roots=(0,),
        feedback=tuple(False for _ in edges),
        edge_weights=tuple(float(index + 1) for index in range(len(edges))),
        weight_semantics="distance_cost",
        required_primitives=frozenset({"nodes", "routes"}),
    )
    graph = replace(
        graph,
        tree_parents=(None, 0, 0, 1, 2, 3, 4, 5),
        tree_depths=(0, 1, 1, 2, 2, 3, 3, 4),
        tree_layout="layered",
        flow_axis=(1.0, 0.0),
        ordered_children={0: (1, 2)},
    )
    return graph, positions


def _ingest_positions(graph: GraphSemantics, positions: torch.Tensor) -> Optional[Scene]:
    """Ingest one drawing with chord routes; ``None`` if rejected."""

    routes = tuple(
        Route(index, torch.stack((positions[source], positions[target])))
        for index, (source, target) in enumerate(graph.edges)
    )
    drawing = DrawingScene(positions, routes, ("nodes", "routes", "node_labels"))
    result = ingest(
        graph,
        drawing,
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes", "node_labels"})),
    )
    if not isinstance(result, ValidScene):
        return None
    return result.scene


def _perturbed_scene(sigma: float, scale: float = 1.0, seed: int = _DEFAULT_SEED) -> Scene:
    """Ingest the semantic fixture under seeded noise and optional sprawl."""

    graph, constructed = _semantic_graph()
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(constructed.shape, generator=generator, dtype=torch.float64)
    scene = _ingest_positions(graph, scale * (constructed + 2.0 * sigma * noise))
    assert scene is not None, "perturbed acceptance fixture failed ingestion"
    return scene


def _exact_subterm(scene: Scene, subterm_id: str, profiles: ScoringProfiles) -> Optional[float]:
    """Evaluate one exact subterm on a scene via the frozen float path."""

    facets = _evaluate_static_facets(scene, profiles)
    result = facets[_SUBTERM_FACET[subterm_id]]
    return result.subterms.get(subterm_id) if result.subterms else None


def test_l_total_backward_end_to_end() -> None:
    """The executed OPUS5 BLOCKER-1 failure flips: backward() fills grads.

    At f3cffecd, ``score_v4_soft(...).l_total.backward()`` raised
    ``element 0 of tensors does not require grad``. The traced path must
    make the same call succeed with a finite nonzero position gradient.
    """

    scene = _perturbed_scene(0.15)
    traced = score_scene_soft(scene, _table(), _profiles())
    assert traced.soft.l_total.requires_grad
    traced.soft.l_total.backward()
    assert traced.positions.grad is not None
    assert bool(torch.isfinite(traced.positions.grad).all())
    assert float(torch.linalg.vector_norm(traced.positions.grad)) > 0.0


def _descent_ascent_gate(
    scene: Scene,
    graph: GraphSemantics,
    subterm_id: str,
    profiles: ScoringProfiles,
    traced,
    label: str,
    exact_before: Optional[float],
) -> None:
    """Run the two-sided smallest-step gradient-alignment gate on one row.

    Descent along -grad of the soft subterm must strictly improve the
    exact subterm WITHOUT annihilating it to zero (a knife-edge fixture
    where any displacement zeroes the events passes an improvement-only
    check with an irrelevant gradient); ascent along +grad at the same
    step must strictly worsen it.
    """

    assert subterm_id in traced.traced_subterms, f"{label} not traced"
    tensor = traced.traced_subterms[subterm_id]
    assert tensor.requires_grad, f"{label} carries no graph"
    assert exact_before is not None and exact_before > 0.0, (
        f"{label} has zero defect; fixture invalid"
    )
    (gradient,) = torch.autograd.grad(
        tensor, traced.positions, retain_graph=True, allow_unused=True
    )
    assert gradient is not None, f"{label} gradient does not reach positions"
    gradient_norm = float(torch.linalg.vector_norm(gradient))
    assert gradient_norm > 0.0, f"{label} gradient is exactly zero at the fixture"
    extent = float((scene.positions.max(dim=0).values - scene.positions.min(dim=0).values).max())
    direction = gradient / gradient_norm

    descended = _ingest_positions(graph, scene.positions.detach() - (_STEP * extent) * direction)
    assert descended is not None, f"{label} descent step failed ingestion"
    exact_after_descent = _exact_subterm(descended, subterm_id, profiles)
    assert exact_after_descent is not None
    assert exact_after_descent < exact_before, (
        f"-grad of soft {label} did not improve the exact facet at the smallest "
        f"step ({exact_before} -> {exact_after_descent})"
    )
    assert exact_after_descent > 0.0, (
        f"{label} annihilated to exactly zero at the smallest step: knife-edge "
        "fixture, the descent direction is not what is being tested"
    )

    ascended = _ingest_positions(graph, scene.positions.detach() + (_STEP * extent) * direction)
    assert ascended is not None, f"{label} ascent step failed ingestion"
    exact_after_ascent = _exact_subterm(ascended, subterm_id, profiles)
    assert exact_after_ascent is not None
    assert exact_after_ascent > exact_before, (
        f"+grad of soft {label} did not worsen the exact facet "
        f"({exact_before} -> {exact_after_ascent}): the improvement is not "
        "carried by the gradient direction"
    )


@pytest.mark.parametrize("subterm_id,sigma,scale,seed", BATTERY)
def test_descent_improves_exact_facet(
    subterm_id: str, sigma: float, scale: float, seed: int
) -> None:
    """Spec 6.5 gradient sanity, two-sided: -grad improves, +grad worsens."""

    profiles = _profiles()
    scene = _perturbed_scene(sigma, scale, seed)
    facets = _evaluate_static_facets(scene, profiles)
    traced = score_scene_soft(scene, _table(), profiles, exact_facets=facets)
    graph, _ = _semantic_graph()
    result = facets[_SUBTERM_FACET[subterm_id]]
    exact_before = result.subterms.get(subterm_id) if result.subterms else None
    _descent_ascent_gate(scene, graph, subterm_id, profiles, traced, subterm_id, exact_before)


@pytest.mark.slow
def test_descent_improves_exact_facet_on_bank_scene() -> None:
    """The two-sided gate holds for all five mandated facets on a REAL bank scene.

    chain/small's scrambled variant is the one pilot-bank scene that
    carries every mandated row live (U01.headline, U7.base, U11.v,
    U17.1, U21.d_sparse_n). The full n>=5-scenes-per-row evidence,
    including medium/large scenes and U7 random-direction controls, is
    banked offline (p3fix5_bank_descent.py); this keeps one bank case
    permanently red-green.
    """

    from tests.eval.ruler_v4.scene_bank import build_cell

    profiles = _profiles()
    scene = build_cell("chain", "small").scenes[5]
    facets = _evaluate_static_facets(scene, profiles)
    traced = score_scene_soft(scene, _table(), profiles, exact_facets=facets)
    for subterm_id in ("U01.headline", "U7.base", "U11.v", "U17.1", "U21.d_sparse_n"):
        result = facets[_SUBTERM_FACET[subterm_id]]
        exact_before = result.subterms.get(subterm_id) if result.subterms else None
        _descent_ascent_gate(
            scene, scene.graph, subterm_id, profiles, traced, f"bank {subterm_id}", exact_before
        )


def test_liveness_summary_floor() -> None:
    """Publish live/flat/constant split on the fixture and gate the floor."""

    scene = _perturbed_scene(0.15)
    traced = score_scene_soft(scene, _table(), _profiles())
    live: Dict[str, float] = {}
    flat = []
    for subterm_id in traced.bound_subterms:
        tensor = traced.traced_subterms[subterm_id]
        if not tensor.requires_grad:
            flat.append(subterm_id)
            continue
        (gradient,) = torch.autograd.grad(
            tensor, traced.positions, retain_graph=True, allow_unused=True
        )
        norm = float(torch.linalg.vector_norm(gradient)) if gradient is not None else 0.0
        if norm > 0.0:
            live[subterm_id] = norm
        else:
            flat.append(subterm_id)
    total_active = len(traced.bound_subterms) + len(traced.constant_subterms)
    assert total_active >= 40, "acceptance fixture lost active rows"
    # Floor: at least half of the fixture's active rows carry live position
    # gradients. The exact split per row is published by the classification
    # sweep; this gate only has to catch a connectivity regression.
    assert len(live) >= total_active // 2, (
        f"live rows {len(live)}/{total_active}; flat={sorted(flat)}; "
        f"constant={sorted(traced.constant_subterms)}"
    )
