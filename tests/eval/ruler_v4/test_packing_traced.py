"""Traced-path gradients for the weight and packing facet families.

Each test proves two properties of the spec 6.5 surrogate seam for the
U35/U36/U38/U41/U42 chains: the traced execution produces live position
gradients for the contract-smoothed closed forms, and the exact float path
is unchanged (an un-traced evaluation of the rebuilt scene is bit-identical
to the original scene, and the traced value agrees with the exact value up
to accumulation order only).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Optional, Tuple

import pytest
import torch

from dagua.eval.ruler_v4._tracing import trace_subterms
from dagua.eval.ruler_v4.composition import CompositionFamily, CompositionProfile
from dagua.eval.ruler_v4.contracts import CONTRACTS
from dagua.eval.ruler_v4.headline import HeadlineProfile
from dagua.eval.ruler_v4.ingestion import ingest
from dagua.eval.ruler_v4.packing import U38, U41, U42
from dagua.eval.ruler_v4.scene import (
    ChannelDeclaration,
    DrawingScene,
    GraphSemantics,
    ObservationProfile,
    Route,
    Scene,
    StyleContract,
    ValidScene,
)
from dagua.eval.ruler_v4.score import ScoringProfiles
from dagua.eval.ruler_v4.surrogate.traced import build_traced_scene, score_scene_soft
from dagua.eval.ruler_v4.weight_table import (
    GATE_DIAGNOSTIC_FACETS,
    REQUIRED_PRIOR_FLOOR_FACETS,
    ParameterProvenance,
    SubtermWeight,
    WeightTable,
)
from dagua.eval.ruler_v4.weights import U35, U36


def _semantic_scene() -> Scene:
    """Build the probe's semantic fixture: a weighted, ranked 8-node DAG.

    Returns
    -------
    Scene
        Validated scene on which U35 and U36 are applicable.
    """

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
    routes = tuple(
        Route(index, torch.stack((positions[source], positions[target])))
        for index, (source, target) in enumerate(edges)
    )
    drawing = DrawingScene(positions, routes, ("nodes", "routes", "node_labels"))
    result = ingest(
        graph,
        drawing,
        StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes", "node_labels"})),
    )
    assert isinstance(result, ValidScene)
    return result.scene


def _scene(
    positions: torch.Tensor,
    edges: Tuple[Tuple[int, int], ...],
    graph_options: Optional[Mapping[str, Any]] = None,
    style: Optional[StyleContract] = None,
) -> Scene:
    """Ingest one small worked-example scene.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    edges : tuple[tuple[int, int], ...]
        Canonical graph edges.
    graph_options : mapping[str, Any] or None
        Optional GraphSemantics field overrides.
    style : StyleContract or None
        Optional corpus-owned style contract.

    Returns
    -------
    Scene
        Validated routed scene.
    """

    graph = GraphSemantics(
        tuple(f"n{index}" for index in range(positions.shape[0])),
        edges,
        **dict(graph_options or {}),
    )
    routes = tuple(
        Route(index, torch.stack((positions[source], positions[target])))
        for index, (source, target) in enumerate(edges)
    )
    result = ingest(
        graph,
        DrawingScene(positions.to(torch.float64), routes),
        style or StyleContract(),
        ObservationProfile(visible_channels=frozenset({"nodes", "routes"})),
    )
    assert isinstance(result, ValidScene)
    return result.scene


def _complete_table() -> WeightTable:
    """Return the probe's complete explicit weight table.

    Returns
    -------
    WeightTable
        One entry per scored manifest sub-term.
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
    """Return the probe's frozen scoring profiles.

    Returns
    -------
    ScoringProfiles
        Complete provenance-audited profile set.
    """

    return ScoringProfiles(
        composition=CompositionProfile(CompositionFamily.P_MEAN, power=2.0),
        headline=HeadlineProfile(index_span=100.0, loss_scale=1.0, version="probe"),
        measurement_version="probe-measurement",
        policy_version="probe-policy",
        alpha_grid_index=1,
        parameter_provenance={
            "composition.power": ParameterProvenance("preregistered_prior"),
            "headline.index_span": ParameterProvenance("contract_frozen"),
            "headline.loss_scale": ParameterProvenance("preregistered_prior"),
            "alpha_grid_index": ParameterProvenance("preregistered_prior"),
        },
    )


def _grad_norm(tensor: torch.Tensor, positions: torch.Tensor) -> float:
    """Return the position-gradient norm of one traced subterm tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Traced scalar subterm.
    positions : torch.Tensor
        Position leaf the trace differentiates against.

    Returns
    -------
    float
        Euclidean gradient norm; zero when the graph never reaches the leaf.
    """

    if not tensor.requires_grad:
        return 0.0
    (gradient,) = torch.autograd.grad(tensor, positions, retain_graph=True, allow_unused=True)
    if gradient is None:
        return 0.0
    assert bool(torch.isfinite(gradient).all())
    return float(torch.linalg.vector_norm(gradient))


def test_weight_facets_trace_live_through_score_scene_soft() -> None:
    """U35/U36 headlines bind live tensors with nonzero position gradients."""

    scene = _semantic_scene()
    traced = score_scene_soft(scene, _complete_table(), _profiles())
    for subterm_id, direct in (("U35.headline", U35(scene)), ("U36.headline", U36(scene))):
        assert subterm_id in traced.bound_subterms
        tensor = traced.traced_subterms[subterm_id]
        assert _grad_norm(tensor, traced.positions) > 0.0
        # Traced forwards may differ from exact by accumulation order only.
        assert direct.value == pytest.approx(float(tensor.detach()), abs=1e-12)
    assert traced.soft.l_total.requires_grad
    (gradient,) = torch.autograd.grad(traced.soft.l_total, traced.positions, retain_graph=True)
    assert bool(torch.isfinite(gradient).all())
    assert float(torch.linalg.vector_norm(gradient)) > 0.0


def test_weight_facets_exact_path_is_bit_identical_off_trace() -> None:
    """The rebuilt scene scores bit-identically outside a trace."""

    scene = _semantic_scene()
    leaf = scene.positions.detach().clone().requires_grad_(True)
    rebuilt = build_traced_scene(scene, leaf)
    for facet in (U35, U36):
        original = facet(scene)
        relinked = facet(rebuilt)
        assert relinked.value == original.value
        assert relinked.subterms == original.subterms


def test_u38_clearance_row_traces_live_on_two_components() -> None:
    """U38's contract-smoothed clearance sigmoid rides the component boxes."""

    scene = _scene(
        torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.2, 0.0], [1.2, 1.0]]),
        ((0, 1), (2, 3)),
    )
    direct = U38(scene)
    assert direct.value is not None
    leaf = scene.positions.detach().clone().requires_grad_(True)
    rebuilt = build_traced_scene(scene, leaf)
    with trace_subterms() as buffer:
        traced_result = U38(rebuilt)
    assert _grad_norm(buffer["U38.L_clear"], leaf) > 0.0
    # L_pack's raster numerator is irreducibly discrete; the live channel is
    # the robust-frame denominator inside the contract's log-sigmoid band.
    assert buffer["U38.L_pack"].requires_grad
    for key, value in direct.subterms.items():
        assert traced_result.subterms[key] == pytest.approx(value, abs=1e-12)
    off_trace = U38(rebuilt)
    assert off_trace.value == direct.value
    assert off_trace.subterms == direct.subterms


def test_u41_face_rows_trace_live_on_certified_triangle() -> None:
    """U41's area-balance burden rides the live arrangement face vertices."""

    height = 3.0**0.5
    scene = _scene(
        torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, height]], dtype=torch.float64),
        ((0, 1), (1, 2), (2, 0)),
        {"planarity_certificate": {"planar": True}},
    )
    direct = U41(scene)
    assert direct.value is not None
    leaf = scene.positions.detach().clone().requires_grad_(True)
    rebuilt = build_traced_scene(scene, leaf)
    with trace_subterms() as buffer:
        traced_result = U41(rebuilt)
    assert _grad_norm(buffer["U41.L_area"], leaf) > 0.0
    assert buffer["U41.L_conv"].requires_grad
    for key, value in direct.subterms.items():
        assert traced_result.subterms[key] == pytest.approx(value, abs=1e-12)
    off_trace = U41(rebuilt)
    assert off_trace.value == direct.value
    assert off_trace.subterms == direct.subterms


def test_u42_proximate_pair_rows_trace_live_through_box_clearance() -> None:
    """U42's C1 proximity gate carries live gradients for box-box pairs.

    The two categories use nearby greys so the dE/20 smoothstep stays inside
    its band (black/white saturates it to an exact-zero constant factor,
    which is genuinely flat); the live channel is the proximity gate.
    """

    declaration = ChannelDeclaration(
        "class_fill",
        "node",
        "class",
        "fill_color",
        {"dark": (0.20, 0.20, 0.20), "dim": (0.30, 0.30, 0.30)},
    )
    graph = GraphSemantics(
        ("n0", "n1"),
        (),
        node_attributes={"class": ("dark", "dim")},
        legends={"class_fill": declaration.value_map},
    )
    ingested = ingest(
        graph,
        DrawingScene(torch.tensor([[0.0, 0.0], [1.5, 0.0]], dtype=torch.float64)),
        StyleContract(channel_set=(declaration,)),
        ObservationProfile(visible_channels=frozenset({"nodes"})),
    )
    assert isinstance(ingested, ValidScene)
    scene = ingested.scene
    direct = U42(scene)
    assert direct.value is not None
    assert "U42.ii" in direct.subterms
    leaf = scene.positions.detach().clone().requires_grad_(True)
    rebuilt = build_traced_scene(scene, leaf)
    with trace_subterms() as buffer:
        traced_result = U42(rebuilt)
    assert _grad_norm(buffer["U42.ii"], leaf) > 0.0
    assert _grad_norm(buffer["U42.iv"], leaf) > 0.0
    for key, value in direct.subterms.items():
        assert traced_result.subterms[key] == pytest.approx(value, abs=1e-12)
    off_trace = U42(rebuilt)
    assert off_trace.value == direct.value
    assert off_trace.subterms == direct.subterms
