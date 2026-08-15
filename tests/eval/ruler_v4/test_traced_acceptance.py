"""Spec 6.5 acceptance battery for the traced differentiable surrogate.

Three gates, all pinned by V4_SPEC_r4 6.5 and the P3 reviews:

1. ``l_total.backward()`` works end to end on the semantic fixture and
   fills a finite nonzero position gradient -- the executed flip of
   P3REVIEW OPUS5 BLOCKER-1 (``element 0 of tensors does not require
   grad``) at f3cffecd.
2. Per-facet gradient sanity: for every battery subterm (the load-bearing
   U01/U07/U11/U17/U21 plus breadth), moving positions along the NEGATIVE
   gradient of the SOFT subterm strictly improves the EXACT subterm after
   re-ingestion, at some step on a small geometric ladder.
3. A liveness summary: the fixture's active rows split into live / flat /
   constant, with a floor on the live fraction so a regression that kills
   gradient connectivity fails loudly.
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

# Battery entries: (subterm_id, noise_sigma, homothety_scale) -- sigma
# scales the seeded perturbation so each target has a nonzero exact
# defect AND room to improve; scale sprawls the whole construction
# (scale * (layout + noise)), which is what puts U21's anti-sprawl row
# off its zero plateau. The five mandated load-bearing facets (U01, U07,
# U11, U17, U21) are all represented; the rest add breadth across every
# traced module, including the clusters family.
BATTERY: Tuple[Tuple[str, float, float], ...] = (
    ("U01.headline", 0.35, 1.0),
    ("U7.base", 0.65, 1.0),
    ("U11.v", 0.35, 1.0),
    ("U17.1", 0.35, 1.0),
    ("U18.le", 0.8, 1.0),
    ("U21.d_sparse_n", 0.35, 4.0),
    ("U03.r_1", 0.35, 1.0),
    ("U09.headline", 0.35, 1.0),
    ("U12.headline", 0.35, 1.0),
    ("U26.i", 0.35, 1.0),
    ("U27.i", 0.35, 1.0),
    ("U28.iii", 0.35, 1.0),
    ("U31.headline", 0.35, 1.0),
    ("U32.L_iso", 0.35, 1.0),
    ("U34.L_mono", 0.35, 1.0),
    ("U35.headline", 0.35, 1.0),
)

_STEP_LADDER = (0.001, 0.005, 0.02, 0.08)


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


def _perturbed_scene(sigma: float, scale: float = 1.0, seed: int = 20260815) -> Scene:
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


@pytest.mark.parametrize("subterm_id,sigma,scale", BATTERY)
def test_descent_improves_exact_facet(subterm_id: str, sigma: float, scale: float) -> None:
    """Spec 6.5 gradient sanity: -grad of the soft subterm improves the exact one."""

    profiles = _profiles()
    scene = _perturbed_scene(sigma, scale)
    traced = score_scene_soft(scene, _table(), profiles)
    assert subterm_id in traced.traced_subterms, f"{subterm_id} not traced"
    tensor = traced.traced_subterms[subterm_id]
    assert tensor.requires_grad, f"{subterm_id} carries no graph"
    exact_before = _exact_subterm(scene, subterm_id, profiles)
    assert exact_before is not None and exact_before > 0.0, (
        f"{subterm_id} has zero defect on the acceptance fixture; battery fixture invalid"
    )
    (gradient,) = torch.autograd.grad(tensor, traced.positions, allow_unused=True)
    assert gradient is not None, f"{subterm_id} gradient does not reach positions"
    gradient_norm = float(torch.linalg.vector_norm(gradient))
    assert gradient_norm > 0.0, f"{subterm_id} gradient is exactly zero at the fixture"
    extent = float((scene.positions.max(dim=0).values - scene.positions.min(dim=0).values).max())
    direction = gradient / gradient_norm
    graph, _ = _semantic_graph()
    improved = False
    for step in _STEP_LADDER:
        stepped = scene.positions.detach() - (step * extent) * direction
        candidate = _ingest_positions(graph, stepped)
        if candidate is None:
            continue
        exact_after = _exact_subterm(candidate, subterm_id, profiles)
        if exact_after is not None and exact_after < exact_before:
            improved = True
            break
    assert improved, (
        f"no step in {_STEP_LADDER} along -grad of soft {subterm_id} improved the "
        f"exact facet from {exact_before}"
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
