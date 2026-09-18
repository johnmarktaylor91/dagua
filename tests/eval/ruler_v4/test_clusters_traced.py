"""Traced-path gradients for the declared-cluster facet family.

Each test proves two properties of the spec 6.5 surrogate seam for the
U25-U30 chains: the traced execution produces live position gradients for
the contract closed forms (region geometry derived from node positions is
live; cluster-label boxes stay input-owned constants), and the exact float
path is unchanged (an un-traced evaluation of the rebuilt scene is
bit-identical to the original scene, and the traced value agrees with the
exact value up to accumulation order only).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Mapping, Optional, Tuple

import pytest
import torch

from dagua.eval.ruler_v4._tracing import trace_subterms
from dagua.eval.ruler_v4.clusters import (
    U25,
    U26,
    U27,
    U28,
    U29,
    U30,
    _segment_circle_interval,
)
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
from dagua.eval.ruler_v4.score import ScoringProfiles
from dagua.eval.ruler_v4.surrogate.traced import build_traced_scene, score_scene_soft
from dagua.eval.ruler_v4.weight_table import (
    GATE_DIAGNOSTIC_FACETS,
    REQUIRED_PRIOR_FLOOR_FACETS,
    ParameterProvenance,
    SubtermWeight,
    WeightTable,
)


def _scene(
    positions: torch.Tensor,
    clusters: Mapping[str, Tuple[int, ...]],
    parents: Optional[Mapping[str, str]] = None,
    cluster_labels_visible: bool = False,
) -> Scene:
    """Ingest one clustered chain-graph scene.

    Parameters
    ----------
    positions : torch.Tensor
        Node positions with shape ``[N, 2]``.
    clusters : mapping[str, tuple[int, ...]]
        Declared cluster memberships.
    parents : mapping[str, str] or None
        Optional child-to-parent hierarchy.
    cluster_labels_visible : bool
        Whether the profile declares cluster-label rendering.

    Returns
    -------
    Scene
        Validated clustered scene.
    """

    edges = tuple((index, index + 1) for index in range(positions.shape[0] - 1))
    graph = GraphSemantics(
        tuple(f"n{index}" for index in range(positions.shape[0])),
        edges,
        clusters=clusters,
        cluster_parents=dict(parents or {}),
    )
    routes = tuple(
        Route(index, torch.stack((positions[source], positions[target])))
        for index, (source, target) in enumerate(edges)
    )
    channels = {"nodes", "routes"}
    if cluster_labels_visible:
        channels.add("cluster_labels")
    result = ingest(
        graph,
        DrawingScene(positions.to(torch.float64), routes),
        StyleContract(),
        ObservationProfile(visible_channels=frozenset(channels)),
    )
    assert isinstance(result, ValidScene)
    return result.scene


def _labelled_hierarchy_scene() -> Scene:
    """Return the probe scene where all six cluster facets are applicable."""

    return _scene(
        torch.tensor(
            [
                [0.0, 0.0],
                [2.0, 0.3],
                [1.0, 1.2],
                [0.4, 2.4],
                [2.2, 2.1],
                [1.3, 3.0],
                [0.9, 1.9],
            ]
        ),
        {"c": (0, 1, 2), "d": (3, 4, 5), "e": (0, 1, 2, 3, 4, 5, 6)},
        {"c": "e", "d": "e"},
        cluster_labels_visible=True,
    )


def _semantic_scene() -> Scene:
    """Build the acceptance semantic fixture: a clustered, ranked 8-node DAG."""

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
    generator = torch.Generator().manual_seed(20260815)
    noise = torch.randn(positions.shape, generator=generator, dtype=torch.float64)
    positions = positions + 0.3 * noise
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


def _complete_table() -> WeightTable:
    """Return the probe's complete explicit weight table."""

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
    """Return the probe's frozen scoring profiles."""

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


def test_cluster_facets_exact_path_is_bit_identical_off_trace() -> None:
    """The rebuilt scene scores bit-identically outside a trace."""

    scene = _labelled_hierarchy_scene()
    leaf = scene.positions.detach().clone().requires_grad_(True)
    rebuilt = build_traced_scene(scene, leaf)
    for facet in (U25, U26, U29):
        original = facet(scene)
        relinked = facet(rebuilt)
        assert relinked.value == original.value
        assert relinked.subterms == original.subterms
    for facet in (U27, U28, U30):
        original = facet(scene, 1)
        relinked = facet(rebuilt, 1)
        assert relinked.value == original.value
        assert relinked.subterms == original.subterms


def test_cluster_facets_traced_values_agree_with_exact() -> None:
    """Traced forwards match exact values up to accumulation order only."""

    scene = _labelled_hierarchy_scene()
    exact = {}
    for name, facet in (("U25", U25), ("U26", U26), ("U29", U29)):
        exact[name] = facet(scene)
    for name, facet in (("U27", U27), ("U28", U28), ("U30", U30)):
        exact[name] = facet(scene, 1)
    leaf = scene.positions.detach().clone().requires_grad_(True)
    rebuilt = build_traced_scene(scene, leaf)
    with trace_subterms() as buffer:
        traced = {}
        for name, facet in (("U25", U25), ("U26", U26), ("U29", U29)):
            traced[name] = facet(rebuilt)
        for name, facet in (("U27", U27), ("U28", U28), ("U30", U30)):
            traced[name] = facet(rebuilt, 1)
    for name, original in exact.items():
        # Facet values pass through the noisy-OR row operator, whose
        # (1 - v) ** w survival amplifies a one-ULP subterm gap at exact
        # saturation; the surrogate binds subterms, so the row values only
        # need the looser bound.
        assert traced[name].value == pytest.approx(original.value, rel=1e-4, abs=1e-12)
        for key, value in (original.subterms or {}).items():
            assert key in buffer, f"{key} not recorded by the trace"
            assert traced[name].subterms[key] == pytest.approx(value, rel=1e-9, abs=1e-12)


def test_u25_spread_cluster_traces_live() -> None:
    """U25's cohesion defect rides the live member radii and robust frame."""

    coords = [[0.0, 0.0], [14.0, 0.0], [7.0, 10.0]]
    for index in range(9):
        coords.append([6.0 + 0.3 * (index % 3), 1.0 + 0.3 * (index // 3)])
    scene = _scene(torch.tensor(coords, dtype=torch.float64), {"wide": (0, 1, 2)})
    direct = U25(scene)
    assert direct.value is not None and direct.value > 0.0
    leaf = scene.positions.detach().clone().requires_grad_(True)
    rebuilt = build_traced_scene(scene, leaf)
    with trace_subterms() as buffer:
        traced = U25(rebuilt)
    assert traced.value == pytest.approx(direct.value, rel=1e-9)
    assert _grad_norm(buffer["U25.headline"], leaf) > 0.0


def test_u26_rows_trace_live_on_partially_separated_clusters() -> None:
    """U26's incidence, matched-control, and boundary rows all carry gradient.

    The control row's fade runs through the historical float32 tensor
    construction; the traced mirror casts down and back up, so the value
    stays bit-equal to the exact path while the gradient flows.
    """

    generator = torch.Generator().manual_seed(7)
    left = torch.randn(14, 2, generator=generator, dtype=torch.float64) * 1.5
    right = torch.randn(14, 2, generator=generator, dtype=torch.float64) * 1.5
    right[:, 0] += 2.0
    scene = _scene(torch.cat([left, right]), {"a": tuple(range(14)), "b": tuple(range(14, 28))})
    direct = U26(scene)
    assert direct.raw["matched_stratum_count"] >= 1
    assert 0.0 < direct.subterms["U26.ii"] < 1.0
    leaf = scene.positions.detach().clone().requires_grad_(True)
    rebuilt = build_traced_scene(scene, leaf)
    with trace_subterms() as buffer:
        traced = U26(rebuilt)
    for key in ("U26.i", "U26.ii", "U26.iii"):
        assert traced.subterms[key] == pytest.approx(direct.subterms[key], rel=1e-9, abs=1e-12)
    # U26.i saturates to an exact 1.0 on this heavy-overlap fixture (a
    # genuinely flat point of the frozen fade); its live witness is the
    # in-band labelled-hierarchy scene below.
    assert _grad_norm(buffer["U26.ii"], leaf) > 0.0

    in_band = _labelled_hierarchy_scene()
    leaf = in_band.positions.detach().clone().requires_grad_(True)
    rebuilt = build_traced_scene(in_band, leaf)
    with trace_subterms() as buffer:
        U26(rebuilt)
    assert _grad_norm(buffer["U26.i"], leaf) > 0.0
    assert _grad_norm(buffer["U26.iii"], leaf) > 0.0


def test_u27_rows_trace_live() -> None:
    """U27's intrusion, member-outside, and route rows all carry gradient."""

    label_scene = _labelled_hierarchy_scene()
    leaf = label_scene.positions.detach().clone().requires_grad_(True)
    rebuilt = build_traced_scene(label_scene, leaf)
    with trace_subterms() as buffer:
        U27(rebuilt, 1)
    assert _grad_norm(buffer["U27.i"], leaf) > 0.0

    outlier_scene = _scene(
        torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, 0.5],
                [1.5, 0.5],
                [0.5, 5.0],
                [9.0, 0.0],
                [10.0, 0.5],
            ],
            dtype=torch.float64,
        ),
        {"a": (0, 1, 2, 3, 4, 5, 6)},
    )
    direct = U27(outlier_scene, 1)
    assert 0.0 < direct.subterms["U27.ii"] < 1.0
    leaf = outlier_scene.positions.detach().clone().requires_grad_(True)
    rebuilt = build_traced_scene(outlier_scene, leaf)
    with trace_subterms() as buffer:
        traced = U27(rebuilt, 1)
    assert traced.subterms["U27.ii"] == pytest.approx(direct.subterms["U27.ii"], rel=1e-9)
    assert _grad_norm(buffer["U27.ii"], leaf) > 0.0

    grazing_scene = _scene(
        torch.tensor(
            [[0.0, 0.0], [2.0, 0.0], [1.0, 1.5], [-10.0, -2.6], [12.0, -2.6]],
            dtype=torch.float64,
        ),
        {"a": (0, 1, 2)},
    )
    direct = U27(grazing_scene, 1)
    assert 0.0 < direct.subterms["U27.iii"] < 1.0
    leaf = grazing_scene.positions.detach().clone().requires_grad_(True)
    rebuilt = build_traced_scene(grazing_scene, leaf)
    with trace_subterms() as buffer:
        traced = U27(rebuilt, 1)
    assert traced.subterms["U27.iii"] == pytest.approx(direct.subterms["U27.iii"], rel=1e-9)
    assert _grad_norm(buffer["U27.iii"], leaf) > 0.0


def test_u28_rows_trace_live() -> None:
    """U28's containment, sibling, and size rows all carry gradient."""

    # A sparse child inside a tightly packed parent: the child's inflation
    # radius exceeds the parent's, so a mild escape sits inside the band.
    coords = [[0.0, 0.0], [0.85, 0.0], [0.425, 0.68]]
    for index in range(10):
        coords.append([3.85 + 0.4 * (index % 4), 1.0 + 0.4 * (index // 4)])
    escape_scene = _scene(
        torch.tensor(coords, dtype=torch.float64),
        {"child": (0, 1, 2), "parent": tuple(range(13))},
        {"child": "parent"},
    )
    direct = U28(escape_scene, 1)
    assert 0.0 < direct.subterms["U28.i"] < 1.0
    leaf = escape_scene.positions.detach().clone().requires_grad_(True)
    rebuilt = build_traced_scene(escape_scene, leaf)
    with trace_subterms() as buffer:
        traced = U28(rebuilt, 1)
    assert traced.subterms["U28.i"] == pytest.approx(direct.subterms["U28.i"], rel=1e-9)
    assert _grad_norm(buffer["U28.i"], leaf) > 0.0

    sibling_scene = _scene(
        torch.tensor(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [1.0, 1.5],
                [6.0, 0.0],
                [8.0, 0.0],
                [7.0, 1.5],
            ],
            dtype=torch.float64,
        ),
        {"c": (0, 1, 2), "d": (3, 4, 5), "e": (0, 1, 2, 3, 4, 5)},
        {"c": "e", "d": "e"},
    )
    direct = U28(sibling_scene, 1)
    assert 0.0 < direct.subterms["U28.ii"] < 1.0
    leaf = sibling_scene.positions.detach().clone().requires_grad_(True)
    rebuilt = build_traced_scene(sibling_scene, leaf)
    with trace_subterms() as buffer:
        traced = U28(rebuilt, 1)
    assert traced.subterms["U28.ii"] == pytest.approx(direct.subterms["U28.ii"], rel=1e-9)
    assert _grad_norm(buffer["U28.ii"], leaf) > 0.0
    assert _grad_norm(buffer["U28.iii"], leaf) > 0.0


def test_u29_ring_cluster_traces_live() -> None:
    """U29's shape quotient rides the live analytic disc-union integrals."""

    count = 49
    ring_radius = 2.0 / (2.0 * math.sin(math.pi / count))
    coords = []
    member_indices = []
    for index in range(count):
        theta = 2.0 * math.pi * index / count
        member_indices.append(len(coords))
        coords.append([ring_radius * math.cos(theta), ring_radius * math.sin(theta)])
        coords.append([40.0 + 0.5 * (index % 5), 40.0 + 0.5 * (index // 5)])
    scene = _scene(torch.tensor(coords, dtype=torch.float64), {"ring": tuple(member_indices)})
    direct = U29(scene)
    assert direct.value is not None and direct.value > 0.0
    leaf = scene.positions.detach().clone().requires_grad_(True)
    rebuilt = build_traced_scene(scene, leaf)
    with trace_subterms() as buffer:
        traced = U29(rebuilt)
    assert traced.value == pytest.approx(direct.value, rel=1e-9)
    assert _grad_norm(buffer["U29.headline"], leaf) > 0.0


def test_u30_rows_trace_live_through_constant_label_boxes() -> None:
    """U30's association and occlusion rows carry gradient via live regions.

    Cluster-label boxes are input-owned constants in this version; the live
    channel is the region geometry (association margins, occlusion budgets).
    The padding row is an anchored zero here: ingestion derives the label at
    the declared padding, so the deviation is exactly zero.
    """

    scene = _labelled_hierarchy_scene()
    direct = U30(scene, 1)
    leaf = scene.positions.detach().clone().requires_grad_(True)
    rebuilt = build_traced_scene(scene, leaf)
    with trace_subterms() as buffer:
        traced = U30(rebuilt, 1)
    for key, value in direct.subterms.items():
        assert traced.subterms[key] == pytest.approx(value, rel=1e-9, abs=1e-12)
    assert _grad_norm(buffer["U30.i"], leaf) > 0.0
    assert _grad_norm(buffer["U30.iii"], leaf) > 0.0
    assert float(buffer["U30.ii"].detach()) == direct.subterms["U30.ii"]
    assert _grad_norm(buffer["U30.ii"], leaf) == 0.0


def test_cluster_rows_bind_live_through_score_scene_soft() -> None:
    """Cluster rows bind live tensors end to end on the semantic fixture."""

    scene = _semantic_scene()
    traced = score_scene_soft(scene, _complete_table(), _profiles())
    for subterm_id in ("U26.i", "U27.i", "U28.iii"):
        assert subterm_id in traced.bound_subterms
        assert _grad_norm(traced.traced_subterms[subterm_id], traced.positions) > 0.0
    assert traced.soft.l_total.requires_grad
    (gradient,) = torch.autograd.grad(traced.soft.l_total, traced.positions, retain_graph=True)
    assert bool(torch.isfinite(gradient).all())
    assert float(torch.linalg.vector_norm(gradient)) > 0.0


def test_coincident_members_keep_exact_values_and_finite_gradients() -> None:
    """Exact-zero member distances take the detached arm without NaN.

    Coincident members hit ``vector_norm`` at the zero vector (undefined
    gradient); the seam's zero-guard returns the value-identical constant, so
    the exact value is unchanged and every traced backward stays finite.
    """

    positions = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [1.0, 0.5],
            [2.0, 0.0],
            [1.5, 1.5],
            [7.0, 0.2],
            [8.0, 0.4],
        ],
        dtype=torch.float64,
    )
    scene = _scene(positions, {"a": (0, 1, 2, 3, 4)})
    exact = {"U25": U25(scene), "U26": U26(scene), "U29": U29(scene)}
    leaf = scene.positions.detach().clone().requires_grad_(True)
    rebuilt = build_traced_scene(scene, leaf)
    with trace_subterms() as buffer:
        traced = {"U25": U25(rebuilt), "U26": U26(rebuilt), "U29": U29(rebuilt)}
    for name, original in exact.items():
        assert traced[name].value == pytest.approx(original.value, rel=1e-9, abs=1e-12)
    for key, tensor in buffer.items():
        if not key.startswith(("U25", "U26", "U29")):
            continue
        _grad_norm(tensor, leaf)  # asserts every gradient is finite


def test_segment_circle_tangency_keeps_value_and_finite_gradient() -> None:
    """Exact tangency (zero discriminant) takes the detached sqrt arm.

    The historical float value at tangency is the degenerate interval
    ``(0.5, 0.5)``; the traced branch must reproduce it without the sqrt
    backward's NaN at zero.
    """

    start = torch.tensor([-1.0, 0.0], dtype=torch.float64)
    end = torch.tensor([1.0, 0.0], dtype=torch.float64)
    center = torch.tensor([0.0, 1.0], dtype=torch.float64)
    exact = _segment_circle_interval(start, end, center, 1.0)
    assert exact == (0.5, 0.5)
    leaf = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    with trace_subterms():
        traced = _segment_circle_interval(leaf[0], leaf[1], center, 1.0)
    assert traced is not None
    entry, exit_ = traced
    assert float(entry.detach()) == 0.5
    assert float(exit_.detach()) == 0.5
    (gradient,) = torch.autograd.grad(entry + exit_, leaf, allow_unused=True)
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
