"""Regression tests for Graphviz fdp compound-edge port handling."""

from __future__ import annotations

import pytest
import torch

from dagua.layout.ops.cluster_geometry import ClusterTree
from dagua.layout.ops.pipelines.fmmm import (
    _FDP_COMPOUND_EDGE_ATTACHMENTS_KEY,
    _fdp_compound_obstacle_list,
    _fdp_compute_compound_edge_attachments,
    _fdp_deepest_cluster_by_node,
    _fdp_expand_box,
    _fdp_node_boxes,
    _fdp_obstacle_vertices,
    _FdpCompoundEdgeAttachmentOp,
    build_fmmm_pipeline,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


def _nested_port_tree() -> ClusterTree:
    """Build a nested cluster tree used by fdp object-list tests.

    Returns
    -------
    ClusterTree
        Tree with ``outer`` containing ``inner`` and root sibling ``sibling``.
    """
    return ClusterTree.from_flat_membership(
        clusters={
            "outer": [0, 1, 2],
            "inner": [1, 2],
            "sibling": [4],
        },
        cluster_parents={
            "outer": None,
            "inner": "outer",
            "sibling": None,
        },
    )


def test_fdp_make_cluster_obstacle_additive_matches_graphviz_golden() -> None:
    """Graphviz ``makeClustObs`` additive expansion should be bit-stable.

    Returns
    -------
    None
        The assertion checks the literal lower-left, upper-left, upper-right,
        lower-right vertex order from ``clusteredges.c``.
    """
    box = _fdp_expand_box(
        key=("cluster", "c"),
        bounds=(10.0, 20.0, 30.0, 50.0),
        expand=(2.0, 3.0),
        do_add=True,
    )

    assert _fdp_obstacle_vertices(box) == (
        (8.0, 17.0),
        (8.0, 53.0),
        (32.0, 53.0),
        (32.0, 17.0),
    )


def test_fdp_make_cluster_obstacle_multiplicative_matches_graphviz_golden() -> None:
    """Graphviz ``makeClustObs`` multiplicative expansion should match C math.

    Returns
    -------
    None
        The assertion validates the branch where ``pm->doAdd`` is false.
    """
    box = _fdp_expand_box(
        key=("cluster", "c"),
        bounds=(10.0, 20.0, 30.0, 50.0),
        expand=(1.5, 2.0),
        do_add=False,
    )

    assert _fdp_obstacle_vertices(box) == (
        (5.0, 5.0),
        (5.0, 65.0),
        (35.0, 65.0),
        (35.0, 5.0),
    )


def test_fdp_object_list_nested_to_sibling_matches_graphviz_walk() -> None:
    """Ported ``objectList`` should exclude endpoint containers by level.

    Returns
    -------
    None
        The expected keys are a golden vector from the C ``raiseLevel`` and
        sibling-walk logic for an edge from ``inner`` to ``sibling``.
    """
    tree = _nested_port_tree()
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
            [40.0, 0.0],
        ],
        dtype=torch.float32,
    )
    node_boxes = _fdp_node_boxes(pos, None, expand=(0.0, 0.0), do_add=True)
    cluster_boxes = {
        "outer": _fdp_expand_box(("cluster", "outer"), (-1.0, -1.0, 21.0, 1.0), (0.0, 0.0), True),
        "inner": _fdp_expand_box(("cluster", "inner"), (9.0, -1.0, 21.0, 1.0), (0.0, 0.0), True),
        "sibling": _fdp_expand_box(
            ("cluster", "sibling"),
            (39.0, -1.0, 41.0, 1.0),
            (0.0, 0.0),
            True,
        ),
    }
    node_parent = _fdp_deepest_cluster_by_node(tree, 5)

    obstacles = _fdp_compound_obstacle_list(
        source=1,
        target=4,
        tree=tree,
        node_parent=node_parent,
        node_boxes=node_boxes,
        cluster_boxes=cluster_boxes,
    )

    assert [obstacle.key for obstacle in obstacles] == [
        ("node", 2),
        ("node", 0),
        ("node", 3),
    ]


def test_fdp_attachment_points_clip_to_crossed_cluster_boundaries() -> None:
    """Inter-cluster edges should attach at source and target cluster boxes.

    Returns
    -------
    None
        The assertion checks deterministic boundary intersections for sibling
        clusters on a horizontal edge.
    """
    problem = LayoutProblem(
        edge_index=torch.tensor([[1], [4]], dtype=torch.long),
        num_nodes=5,
        node_sizes=torch.zeros((5, 2), dtype=torch.float32),
        clusters={
            "outer": [0, 1, 2],
            "inner": [1, 2],
            "sibling": [4],
        },
        cluster_parents={
            "outer": None,
            "inner": "outer",
            "sibling": None,
        },
    )
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
            [40.0, 0.0],
        ],
        dtype=torch.float32,
    )

    attachments, _, _ = _fdp_compute_compound_edge_attachments(problem, pos)

    assert len(attachments) == 1
    attachment = attachments[0]
    assert attachment.tail_cluster == "inner"
    assert attachment.head_cluster == "sibling"
    assert attachment.tail_point == pytest.approx((55.0, 0.0))
    assert attachment.head_point == pytest.approx((5.0, 0.0))
    assert attachment.polyline[0] == pytest.approx((55.0, 0.0))
    assert attachment.polyline[1] == pytest.approx((5.0, 0.0))


def test_fdp_compound_attachment_op_is_fidelity_only() -> None:
    """The fdp attachment op should only be present for ``fidelity_mode``.

    Returns
    -------
    None
        The assertion protects default behavior while checking the fidelity
        pipeline records metadata in ``SolveState.extras``.
    """
    default_pipeline = build_fmmm_pipeline(steps=0)
    fidelity_pipeline = build_fmmm_pipeline(steps=0, fidelity_mode=True)

    assert [op.name for op in default_pipeline.ops].count(_FdpCompoundEdgeAttachmentOp.name) == 0
    assert [op.name for op in fidelity_pipeline.ops].count(_FdpCompoundEdgeAttachmentOp.name) == 1

    problem = LayoutProblem(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        num_nodes=2,
        clusters={"left": [0], "right": [1]},
        cluster_parents={"left": None, "right": None},
    )
    state = SolveState(pos=torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float32))

    result = _FdpCompoundEdgeAttachmentOp().apply(problem, state, RuntimeContext())

    assert _FDP_COMPOUND_EDGE_ATTACHMENTS_KEY in result.extras
    assert result.extras[_FDP_COMPOUND_EDGE_ATTACHMENTS_KEY][0].tail_point == pytest.approx(
        (35.0, 0.0)
    )
