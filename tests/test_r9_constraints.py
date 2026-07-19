"""Regression tests for the R9 user-intent constraint API."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

import dagua
from dagua import C, DaguaGraph, LayoutConfig
from dagua.constraints import (
    ConstraintConflictError,
    ConstraintStagedError,
    ConstraintTypeError,
    resolve_edge_selection,
    resolve_node_selection,
)
from dagua.layout.losses import (
    constraint_anchor_loss,
    constraint_emphasize_loss,
    constraint_separate_loss,
)


def _tiny_graph() -> DaguaGraph:
    """Return a small graph with stable node ids.

    Returns
    -------
    DaguaGraph
        Graph containing ``a``, ``b``, and ``c``.
    """
    graph = DaguaGraph()
    for node_id in ("a", "b", "c"):
        graph.add_node(node_id)
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    return graph


def test_constraint_construction_and_fluent_immutability() -> None:
    """Constraint constructors and fluent methods return immutable copies."""
    constraint = C.Align("a", "b")
    stronger = constraint.hard()

    assert constraint.strength == "rigid"
    assert stronger.strength == "hard"
    assert constraint is not stronger
    with pytest.raises(FrozenInstanceError):
        constraint.axis = "y"  # type: ignore[misc]


def test_selection_resolution_for_bare_list_cluster_where_and_path() -> None:
    """Selections resolve eager ids and lazy node selectors deterministically."""
    graph = _tiny_graph()
    graph.add_cluster("pair", ["a", "b"])

    assert resolve_node_selection("a", graph) == [0]
    assert resolve_node_selection(["a", "c"], graph) == [0, 2]
    assert resolve_node_selection(C.cluster("pair"), graph) == [0, 1]
    assert resolve_node_selection(C.where(lambda node_id: node_id != "b"), graph) == [0, 2]
    assert resolve_node_selection(C.path("a", "b", "c"), graph) == [0, 1, 2]


def test_hard_refusal_for_unprojectable_types() -> None:
    """Unprojectable hard intents raise at construction time."""
    with pytest.raises(ConstraintTypeError, match="hard separate requires a committed side"):
        C.Separate("a", "b", strength="hard")
    with pytest.raises(ConstraintTypeError, match="hard emphasize"):
        C.Emphasize("a", "b", strength="hard")
    with pytest.raises(ConstraintTypeError, match="hard focus"):
        C.Focus("a", strength="hard")
    with pytest.raises(ConstraintTypeError, match='fit="fixed"'):
        C.Anchor({"a": (0.0, 0.0)}, strength="hard")


def test_graph_verbs_aliases_and_report() -> None:
    """Graph verbs append constraints and expose a post-layout report."""
    graph = _tiny_graph()
    pin = graph.pin("a", 0.0, 0.0)
    row = graph.row("b", "c")
    left = graph.left_of("a", "b")

    assert pin in list(graph.constraints)
    assert row.axis == "y"
    assert left.axis == "x"

    pos = dagua.layout(graph, LayoutConfig(steps=2, device="cpu"))
    assert torch.allclose(pos[0], torch.zeros(2), atol=0.0)
    report = graph.constraints.report()
    assert len(report.residuals) == 3
    assert report.residuals[0].hard_satisfied


def test_no_constraints_identity_with_empty_constraint_list() -> None:
    """An empty constraint list leaves Tier-0 layout bytes unchanged."""
    plain = _tiny_graph()
    with_empty = _tiny_graph()
    with_empty.constraints.clear()

    config = LayoutConfig(steps=3, device="cpu", seed=7, algorithm="_legacy")
    plain_pos = dagua.layout(plain, config)
    empty_pos = dagua.layout(with_empty, config)

    assert torch.equal(plain_pos, empty_pos)


def test_element_selectors_lower_for_edges_labels_canvas_and_contain() -> None:
    """R9 element selectors resolve honestly for edges, labels, and canvas."""
    graph = _tiny_graph()
    graph.edge_types[0] = "critical"

    graph.emphasize(C.edges(source="a", type="critical"))
    graph.pin(C.label("a"), at=C.canvas.center)
    graph.constrain(C.Contain(["a", "b"], within=C.canvas, padding=4, strength="firm"))

    pos = dagua.layout(graph, LayoutConfig(steps=2, device="cpu", constraint_policy="report"))

    assert pos.shape == (3, 2)
    report = graph.constraints.report()
    assert len(report.residuals) == 3
    assert not [row for row in report.violations if row.constraint.is_hard]


def test_port_constraints_raise_staged_error_on_lowering() -> None:
    """Port selectors are API-fixed but lowering is explicitly staged."""
    graph = _tiny_graph()
    graph.pin(C.port("a", "east"))

    with pytest.raises(ConstraintStagedError, match="staged to the edge-routing sprint"):
        dagua.layout(graph, LayoutConfig(steps=1, device="cpu"))


def test_constraint_report_accepts_positions_and_marks_violations() -> None:
    """ConstraintSet.report(pos=...) computes normalized residuals on demand."""
    graph = _tiny_graph()
    graph.pin("a", x=10.0, y=0.0, name="pin-a")

    report = graph.constraints.report(torch.zeros(3, 2))

    assert report.residuals[0].residual == 10.0
    assert report.residuals[0].resolved_count == 1
    assert report.violations[0].constraint.name == "pin-a"


def test_strict_constraint_policy_raises_for_hard_violation() -> None:
    """Strict policy raises when post-layout hard residuals remain violated."""
    graph = _tiny_graph()
    graph.pin("a", x=0.0, y=0.0)
    graph.pin("a", x=10.0, y=0.0)

    with pytest.raises(ConstraintConflictError):
        dagua.layout(graph, LayoutConfig(steps=1, device="cpu", constraint_policy="strict"))


def test_constraints_polish_explicit_pipeline_algorithm() -> None:
    """R9 constraints apply after explicit non-default pipeline algorithms."""
    graph = _tiny_graph()
    graph.pin("a", x=123.0, y=-45.0)

    pos = dagua.layout(graph, LayoutConfig(algorithm="fr", steps=2, device="cpu"))

    assert torch.allclose(pos[0], torch.tensor([123.0, -45.0]), atol=0.0)


def test_constraints_round_trip_through_graph_json() -> None:
    """Built-in constraints serialize and load through graph JSON."""
    graph = _tiny_graph()
    graph.pin("a", x=1.0, y=2.0, name="origin")
    graph.constrain(C.Contain(["b", "c"], within=C.canvas, padding=3.0, strength="hard"))

    data = dagua.graph_to_json(graph)
    loaded = dagua.graph_from_json(data)

    assert len(loaded.constraints) == 2
    assert loaded.constraints[0].name == "origin"
    assert isinstance(loaded.constraints[1], C.Contain)


def test_canonical_one_liner_example_runs() -> None:
    """The canonical Tier 0/1 one-liner example executes."""
    graph = DaguaGraph()
    for node_id in ("input", "search", "reason", "safety", "extract", "transform", "load"):
        graph.add_node(node_id)
    graph.add_edge("extract", "transform")
    graph.add_edge("transform", "load")

    graph.pin("input", 0, 0)
    graph.row("search", "reason", "safety")
    graph.order("extract", "transform", "load")
    graph.separate("search", "load", gap=20)
    pos = dagua.layout(graph, LayoutConfig(steps=2, device="cpu"))

    assert pos.shape == (7, 2)


def test_canonical_mid_level_example_runs() -> None:
    """The canonical selector-rich service-map example executes."""
    graph = DaguaGraph()
    for dc in ("nyc", "chi", "sf"):
        for role in ("lb0", "app0", "db0"):
            graph.add_node(f"{dc}/{role}")
        graph.add_edge(f"{dc}/lb0", f"{dc}/app0")
        graph.add_edge(f"{dc}/app0", f"{dc}/db0")
    graph.add_node("legend")
    graph.add_edge("nyc/lb0", "chi/lb0", label="failover")

    graph.anchor({"nyc/lb0": (-74.0, 40.7), "chi/lb0": (-87.6, 41.9), "sf/lb0": (-122.4, 37.8)})
    for dc in ("nyc", "chi", "sf"):
        graph.order(
            C.where(lambda node_id, d=dc: str(node_id).startswith(f"{d}/lb")),
            C.where(lambda node_id, d=dc: str(node_id).startswith(f"{d}/app")),
            C.where(lambda node_id, d=dc: str(node_id).startswith(f"{d}/db")),
            axis="flow",
        )
    graph.emphasize("nyc/lb0", "chi/lb0", "chi/app0")
    graph.pin("legend", at=C.canvas.fraction(0.95, 0.05))
    graph.separate(C.labels(edges=True), C.canvas.edge("bottom"), gap=20)
    pos = dagua.layout(graph, LayoutConfig(steps=2, device="cpu", constraint_policy="report"))

    assert pos.shape == (10, 2)
    assert "constraints" in graph.constraints.report().summary()


def test_canonical_power_user_escape_hatch_runs() -> None:
    """The canonical headless power-user escape hatches execute."""
    graph = DaguaGraph()
    for node_id in ("core", "s0", "s1", "legend", "scale"):
        graph.add_node(node_id)
    for spoke in ("s0", "s1"):
        graph.add_edge("core", spoke)
    graph.add_cluster("pay", ["s0"])
    graph.add_cluster("search", ["s1"])

    def snap_core(pos: torch.Tensor, ctx: C.ConstraintContext) -> None:
        """Snap the core node exactly to the origin."""
        idx = ctx.idx("core")
        pos[idx] = torch.zeros(2, dtype=pos.dtype, device=pos.device)

    cfg = LayoutConfig(
        algorithm="stress_sgd",
        steps=2,
        device="cpu",
        constraints=[
            C.Pin("core", x=0, y=0),
            C.project(snap_core, name="snap-core"),
            C.Separate(C.cluster("pay"), C.cluster("search"), gap=12).rigid(),
            C.Contain(["legend", "scale"], within=C.canvas, padding=12),
            C.loss(
                lambda pos, ctx: pos[ctx.indices(["s0", "s1"]), 0].var(),
                strength="soft",
                name="leaf-spread",
            ),
        ],
        constraint_policy="report",
    )
    pos = dagua.layout(graph, cfg)

    assert torch.allclose(pos[0], torch.zeros(2), atol=0.0)


def test_hard_pin_and_flow_order_are_exact_under_each_direction() -> None:
    """Hard view-frame constraints are exact after direction handling."""
    for direction in ("TB", "BT", "LR", "RL"):
        graph = _tiny_graph()
        graph.direction = direction
        graph.pin("a", x=100.0, y=50.0)
        graph.order("c", "b", axis="flow", gap=25.0)

        pos = dagua.layout(graph, LayoutConfig(steps=2, device="cpu"))
        axis = 0 if direction in {"LR", "RL"} else 1

        assert torch.allclose(pos[0], torch.tensor([100.0, 50.0]), atol=0.0)
        assert float(pos[2, axis] + 25.0 - pos[1, axis]) <= 0.0
        assert not graph.constraints.report().violations


def test_graph_and_config_constraints_union_and_config_strict_policy() -> None:
    """Graph constraints merge with config constraints and strict covers both."""
    graph = _tiny_graph()
    graph.pin("a", x=1.0, y=2.0)
    config = LayoutConfig(
        algorithm="fr",
        steps=2,
        device="cpu",
        constraints=[C.Pin("b", x=3.0, y=4.0)],
        constraint_policy="strict",
    )

    pos = dagua.layout(graph, config)

    assert torch.allclose(pos[0], torch.tensor([1.0, 2.0]), atol=0.0)
    assert torch.allclose(pos[1], torch.tensor([3.0, 4.0]), atol=0.0)
    assert len(graph.constraints.report().residuals) == 2

    contradictory = _tiny_graph()
    with pytest.raises(ConstraintConflictError):
        dagua.layout(
            contradictory,
            LayoutConfig(
                algorithm="fr",
                steps=1,
                device="cpu",
                constraints=[C.Pin("a", x=0.0), C.Pin("a", x=10.0)],
                constraint_policy="strict",
            ),
        )


def test_hard_group_and_hard_contain_are_projected_without_nan() -> None:
    """Hard group and Contain use exact projection instead of infinite losses."""
    graph = _tiny_graph()
    graph.group("a", "b", "c", strength="hard")
    graph.constrain(C.Contain(["a", "b"], within=C.canvas, padding=5.0, strength="hard"))

    pos = dagua.layout(graph, LayoutConfig(algorithm="fr", steps=2, device="cpu"))

    assert torch.isfinite(pos).all()
    assert torch.allclose(pos, pos[:1].expand_as(pos), atol=0.0)
    assert not graph.constraints.report().violations


def test_constraint_losses_have_finite_gradients_at_coincident_points() -> None:
    """Distance-based losses keep finite gradients at zero distance."""
    pos = torch.zeros(3, 2, requires_grad=True)
    separate_loss = constraint_separate_loss(
        pos,
        [(torch.tensor([0]), torch.tensor([1]), None, 2.0, 1.0)],
    )
    emphasize_loss = constraint_emphasize_loss(pos, [(torch.tensor([0, 1, 2]), 1.0)])
    loss = separate_loss + emphasize_loss
    loss.backward()

    assert torch.isfinite(pos.grad).all()


def test_anchor_similarity_preserves_shape_and_fixed_anchor_is_exact() -> None:
    """Default anchors fit by similarity while fixed anchors remain absolute."""
    graph = _tiny_graph()
    reference = torch.tensor([[-74.0, 40.7], [-87.6, 41.9], [-122.4, 37.8]])
    graph.anchor({"a": (-74.0, 40.7), "b": (-87.6, 41.9), "c": (-122.4, 37.8)})
    pos = dagua.layout(graph, LayoutConfig(algorithm="fr", steps=10, device="cpu"))

    assert torch.isfinite(pos).all()
    assert float(torch.pdist(pos).max().item()) > 0.0
    indices = torch.tensor([0, 1, 2], dtype=torch.long)
    translated_scaled = reference * 3.0 + torch.tensor([250.0, -80.0])
    similarity_loss = constraint_anchor_loss(
        translated_scaled,
        [(indices, reference, 1.0, "similarity")],
    )
    raw_loss = constraint_anchor_loss(
        translated_scaled,
        [(indices, reference, 1.0, "fixed")],
    )

    assert similarity_loss.item() == pytest.approx(0.0, abs=1.0e-10)
    assert raw_loss.item() > 1.0

    fixed = _tiny_graph()
    fixed.anchor({"a": (7.0, 8.0)}, fit="fixed", strength="hard")
    fixed_pos = dagua.layout(fixed, LayoutConfig(algorithm="fr", steps=2, device="cpu"))

    assert torch.allclose(fixed_pos[0], torch.tensor([7.0, 8.0]), atol=0.0)


def test_label_pin_records_satellite_without_moving_owner() -> None:
    """A constrained label pin is independent of its owner node position."""
    graph = _tiny_graph()
    graph.pin("a", x=0.0, y=0.0)
    graph.pin(C.label("a"), x=42.0, y=24.0)

    pos = dagua.layout(graph, LayoutConfig(algorithm="fr", steps=2, device="cpu"))
    label_positions = getattr(graph, "_r9_label_positions", {})

    assert torch.allclose(pos[0], torch.zeros(2), atol=0.0)
    assert any(value == (42.0, 24.0) for value in label_positions.values())
    assert not graph.constraints.report().violations


def test_edges_filters_intersect_and_dead_kwargs_raise() -> None:
    """Selector filters are conjunctive and unsupported knobs are typed errors."""
    graph = _tiny_graph()
    graph.add_edge("c", "a")
    graph.edge_types[0] = "critical"
    graph.edge_types[2] = "critical"

    assert resolve_edge_selection(C.edges(source="c", type="critical"), graph) == [2]
    with pytest.raises(ConstraintTypeError):
        C.Align("a", "b", spacing=10.0)
    with pytest.raises(ConstraintTypeError):
        C.Focus("a", radius=3)
    with pytest.raises(ConstraintTypeError):
        C.Emphasize("a", "b", lane="main")
    with pytest.raises(ConstraintTypeError):
        C.Pin("a", x=1.0, frame="layout")


def test_matrix_holes_and_project_round_trip_are_honest() -> None:
    """Unsupported matrix cells raise and custom projectors load without crashing."""
    graph = _tiny_graph()

    with pytest.raises(ConstraintTypeError):
        graph.pin(C.path("a", "b"))
    with pytest.raises(ConstraintTypeError):
        graph.order(C.label("a"), "b")
    with pytest.raises(ConstraintTypeError):
        graph.align(C.edges(source="a"), "b")

    def snap(pos: torch.Tensor, ctx: C.ConstraintContext) -> None:
        """No-op projector for serialization coverage."""
        return None

    graph.constrain(C.project(snap, name="snap"))
    loaded = dagua.graph_from_json(dagua.graph_to_json(graph))

    assert len(loaded.constraints) == 1
    assert loaded.constraints[0].name == "snap"
