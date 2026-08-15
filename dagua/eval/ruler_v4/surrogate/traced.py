"""Traced scene scoring: spec 6.5's differentiable surrogate over positions.

Design (recorded per MODULARITY.md): the traced path re-executes the SAME
facet implementations on a rebuilt scene whose geometry is re-linked to a
position leaf carrying ``requires_grad``. Facet arithmetic stays behind
the frozen ``FacetResult`` seam -- tensors reach the surrogate through the
context-scoped trace buffer (``_tracing.trace_subterms``), which
``value_result`` fills, never through a changed result shape. The exact
scorer's outputs are bit-identical to the pre-surrogate implementation
because every polymorphic seam executes the historical float operations
outside a trace. Traced forwards may differ from exact by accumulation
order only; that gap is measured by the certification harness, never
assumed zero.

Gradient channels, stated honestly:

- node positions and every geometry derived from them in the rebuild
  (node boxes, node-label boxes, chord-identical routes) are live;
- producer routes that are NOT chord-identical, edge-label positions, and
  cluster-label boxes are input-owned constants in this version: facets
  reading only those channels contribute zero position gradient;
- constructs that are a.e. piecewise-constant in positions (hard counts,
  order statistics' rank structure, event indicators) have exact zero
  gradient wherever no contract names a smoothing; DISCREPANCIES entry 38
  states the exemption rule and the classification sweep publishes the
  per-row status.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Mapping, Optional, Tuple

import torch

if TYPE_CHECKING:
    from dagua.eval.ruler_v4.score import ScoringProfiles

from dagua.eval.ruler_v4._tracing import trace_subterms
from dagua.eval.ruler_v4.composition import compose
from dagua.eval.ruler_v4.scene import BoxGeometry, Route, Scene, TemporalScene
from dagua.eval.ruler_v4.surrogate.scorer import SoftScoreResult, score_v4_soft
from dagua.eval.ruler_v4.weight_table import WeightTable


@dataclass(frozen=True)
class TracedSoftScore:
    """Publish one traced differentiable scoring of a validated scene.

    Parameters
    ----------
    soft : SoftScoreResult
        Differentiable surrogate outputs; ``soft.l_total.backward()``
        populates ``positions.grad``.
    positions : torch.Tensor
        The ``[N, 2]`` float64 leaf tensor the score differentiates against.
    traced_subterms : mapping[str, torch.Tensor]
        Every score-visible subterm the traced execution produced as a live
        tensor, keyed by manifest id.
    bound_subterms : tuple[str, ...]
        Active rows bound to traced tensors in ``soft``.
    constant_subterms : tuple[str, ...]
        Active rows with no traced tensor, bound as exact-value constants
        (zero gradient by construction; honest, not hidden).
    """

    soft: SoftScoreResult
    positions: torch.Tensor
    traced_subterms: Mapping[str, torch.Tensor]
    bound_subterms: Tuple[str, ...]
    constant_subterms: Tuple[str, ...]


def build_traced_scene(scene: Scene, positions: torch.Tensor) -> Scene:
    """Rebuild a validated scene with geometry re-linked to a position leaf.

    Node boxes ride their owner's live position; node-label boxes keep their
    validated offset from the owner as a constant displacement; a producer
    route that is exactly the chord of its endpoints is re-linked so route
    geometry follows the endpoints. Everything else (non-chord routes,
    edge-label boxes, cluster-label boxes, hashes, intrinsic unit) is carried
    unchanged as input-owned constants.

    Parameters
    ----------
    scene : Scene
        Validated canonical scene.
    positions : torch.Tensor
        Float64 ``[N, 2]`` tensor with the same values as ``scene.positions``
        (typically ``scene.positions.detach().clone().requires_grad_(True)``).

    Returns
    -------
    Scene
        Scene whose derived geometry carries the autograd graph.

    Raises
    ------
    ValueError
        If the position tensor does not match the scene's positions shape.
    """

    if positions.shape != scene.positions.shape:
        raise ValueError("traced positions must match the validated scene shape")
    node_boxes = tuple(
        BoxGeometry(positions[box.owner], box.half_extents, box.owner) for box in scene.node_boxes
    )
    node_label_boxes = tuple(
        BoxGeometry(
            positions[box.owner] + (box.center - scene.positions[box.owner]).detach(),
            box.half_extents,
            box.owner,
        )
        for box in scene.node_label_boxes
    )
    routes = []
    for route in scene.routes:
        source, target = scene.graph.edges[route.edge_index]
        chord = torch.stack((scene.positions[source], scene.positions[target]))
        if route.points.shape == chord.shape and bool(torch.equal(route.points, chord)):
            routes.append(
                Route(
                    edge_index=route.edge_index,
                    points=torch.stack((positions[source], positions[target])),
                    kind=route.kind,
                )
            )
        else:
            routes.append(route)
    return Scene(
        graph=scene.graph,
        style=scene.style,
        profile=scene.profile,
        positions=positions,
        routes=tuple(routes),
        z_order=scene.z_order,
        node_boxes=node_boxes,
        node_label_boxes=node_label_boxes,
        edge_label_boxes=scene.edge_label_boxes,
        cluster_label_boxes=scene.cluster_label_boxes,
        intrinsic_unit=scene.intrinsic_unit,
        profile_hash=scene.profile_hash,
        graph_hash=scene.graph_hash,
    )


def score_scene_soft(
    scene: Scene,
    weight_table: WeightTable,
    profiles: "ScoringProfiles",
    temporal_scene: Optional[TemporalScene] = None,
    *,
    positions: Optional[torch.Tensor] = None,
) -> TracedSoftScore:
    """Score one validated scene with position gradients end to end.

    Runs the exact scorer's facet evaluation twice: once normally (the
    frozen float path, which defines applicability and the exact reference
    composition) and once inside a trace on the rebuilt scene, collecting a
    live tensor per score-visible subterm. Active rows with traced tensors
    are bound into :func:`score_v4_soft`; the remainder stay exact-value
    constants and are published as such.

    Parameters
    ----------
    scene : Scene
        Validated canonical scene.
    weight_table : WeightTable
        Complete explicit per-sub-term weights.
    profiles : ScoringProfiles
        Frozen scoring profiles (composition, facet parameters).
    temporal_scene : TemporalScene or None
        Optional validated temporal history for U40; traced as a constant.
    positions : torch.Tensor or None
        Optional caller-owned position leaf. Defaults to a detached clone of
        the scene's positions with ``requires_grad=True``.

    Returns
    -------
    TracedSoftScore
        Differentiable score and per-term provenance.
    """

    from dagua.eval.ruler_v4.score import ScoringProfiles, _evaluate_static_facets

    assert isinstance(profiles, ScoringProfiles)
    exact_facets = _evaluate_static_facets(scene, profiles, temporal_scene)
    if positions is None:
        positions = scene.positions.detach().clone().requires_grad_(True)
    traced_scene = build_traced_scene(scene, positions)
    buffer: Dict[str, torch.Tensor]
    with trace_subterms() as buffer:
        _evaluate_static_facets(traced_scene, profiles, temporal_scene)
    exact_composition = compose(exact_facets, weight_table, profiles.composition)
    active = tuple(
        row.subterm_id
        for row in exact_composition.subterms
        if row.value is not None and not row.diagnostic and row.normalized_weight > 0.0
    )
    term_tensors = {subterm_id: buffer[subterm_id] for subterm_id in active if subterm_id in buffer}
    soft = score_v4_soft(
        exact_facets,
        weight_table,
        profiles.composition,
        term_tensors=term_tensors,
    )
    return TracedSoftScore(
        soft=soft,
        positions=positions,
        traced_subterms=dict(buffer),
        bound_subterms=tuple(sorted(term_tensors)),
        constant_subterms=tuple(
            sorted(subterm_id for subterm_id in active if subterm_id not in term_tensors)
        ),
    )


def position_gradient_norms(traced: TracedSoftScore) -> Mapping[str, float]:
    """Measure each bound term's position-gradient magnitude.

    Parameters
    ----------
    traced : TracedSoftScore
        Traced scoring result.

    Returns
    -------
    mapping[str, float]
        Euclidean norm of ``d term / d positions`` per bound subterm id;
        exactly ``0.0`` where the term is locally flat, absent where the
        term carries no graph to the position leaf.
    """

    norms = {}
    for subterm_id in traced.bound_subterms:
        tensor = traced.traced_subterms[subterm_id]
        if not tensor.requires_grad:
            continue
        (gradient,) = torch.autograd.grad(
            tensor,
            traced.positions,
            retain_graph=True,
            allow_unused=True,
        )
        if gradient is not None:
            norms[subterm_id] = float(torch.linalg.vector_norm(gradient))
    return norms
