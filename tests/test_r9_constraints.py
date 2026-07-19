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
    resolve_node_selection,
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
    """R9 element selectors lower through the current tensor engine."""
    graph = _tiny_graph()
    graph.edge_types[0] = "critical"

    graph.emphasize(C.edges(source="a", type="critical"))
    graph.pin(C.label("a"), at=C.canvas.center)
    graph.align(C.labels(["b", "c"]), at=C.canvas.edge("top"), axis="y")
    graph.constrain(C.Contain(["a", "b"], within=C.canvas, padding=4, strength="hard"))

    pos = dagua.layout(graph, LayoutConfig(steps=2, device="cpu"))

    assert pos.shape == (3, 2)
    report = graph.constraints.report()
    assert len(report.residuals) == 4


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
